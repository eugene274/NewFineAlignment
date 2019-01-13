#include <utility>

#include <getopt.h>

#include <iostream>
#include <memory>

#include "types.h"
#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooPolynomial.h>
#include <RooDataSet.h>
#include <RooPlot.h>
#include <TCanvas.h>
#include <TText.h>
#include <TStyle.h>
#include <Math/IFunction.h>
#include <Math/GSLMinimizer.h>
#include <TDecompSVD.h>
#include <TPaveText.h>
#include <TH2D.h>
#include <TChain.h>

constexpr int N_SLICE_ENTRIES = 5000;
constexpr int N_SLICES_PRINT = 10;

TpcCalibData gTpcCalibData;
typedef std::array<TpcCalibData, N_SLICE_ENTRIES> SliceData;

class SliceDiscrepancyFCT : public ROOT::Math::IGradientFunctionMultiDim {

 public:
  explicit SliceDiscrepancyFCT(std::shared_ptr<SliceData> _sliceDataPtr) : sliceDataPtr(std::move(_sliceDataPtr)) {}

  unsigned int NDim() const override {
    return 3;
  }

  void FdF(const double *x, double &f, double *df) const override {
    using namespace std;
    using namespace TMath;


    double dPhi = x[0];
    TVectorD U0(2, &x[1]);

    TMatrixD A(2, 2);
    TMatrixD DA(2, 2);
    {
      double si = Sin(dPhi);
      double co = Cos(dPhi);

      double a_arr[] = {
          co, si,
          -si, co};
      double da_arr[] = {
          -si, co,
          -co, -si};
      A.SetMatrixArray(a_arr);
      DA.SetMatrixArray(da_arr);
    }

    double _f = 0;
    TVectorD _df(NDim());

    for (auto data : *sliceDataPtr) {
      auto u = data.u();
      auto up = data.uPrim();

      if ((u - up).Norm1() > 1.)
        continue;

      auto d = (A * u + U0) - up;
      _f += 1. / N_SLICE_ENTRIES * d.Norm2Sqr();

      _df[0] += 2. / N_SLICE_ENTRIES * (d * (DA * u));
      _df[1] += 2. / N_SLICE_ENTRIES * d[0];
      _df[2] += 2. / N_SLICE_ENTRIES * d[1];
    }

    double *dfArr = _df.GetMatrixArray();
    copy(&dfArr[0], &dfArr[NDim()], &df[0]);

    f = _f;

  }

  IBaseFunctionMultiDimTempl<double> *Clone() const override {
    return new SliceDiscrepancyFCT(this->sliceDataPtr);
  }

 private:
  double DoEval(const double *x) const override {
    double f;
    double fdf[NDim()];
    FdF(x, f, fdf);
    return f;
  }

  double DoDerivative(const double *x, unsigned int icoord) const override {
    double f;
    double fdf[NDim()];
    FdF(x, f, fdf);
    return fdf[icoord];
  }

 private:
  std::shared_ptr<SliceData> sliceDataPtr;
};

TPaveText *GenerateSliceInfo(const SliceAnalysisData &stat) {

  auto matrix = [](const TMatrixD &m) {
    return Form("#left( #splitline{%5.3f  %5.3f}{%5.3f  %5.3f}  #right)",
                m[0][0], m[0][1],
                m[1][0], m[1][1]);
  };

  auto vector = [](const TVectorD &m) {
    return Form("(%5.3f; %5.3f)", m[0], m[1]);
  };

  auto pave = new TPaveText(0.1, 0.1, 0.9, 0.9);
  pave->SetTextSize(0.02);
  pave->SetTextAlign(kHAlignLeft + kVAlignTop);
  pave->SetMargin(0.1);

  const double gNewLineLarge = -0.05;
  const double gNewLineSmall = -0.03;
  double YY = 0.8;

  pave->AddText(0.1, YY,
                Form("Slice %d: Events %u - %u",
                     stat.sliceID,
                     stat.sliceEventStart,
                     stat.sliceEventEnd))->SetTextSize(0.04);
  YY += gNewLineLarge;
  pave->AddText(0.1, YY, Form("#Delta T = %u s", stat.sliceUnixTimeEnd - stat.sliceUnixTimeStart));
  YY += gNewLineLarge;
  pave->AddLine(0.1, YY, 0.9, YY);
  YY += gNewLineSmall;
  if (stat.minimizationIsSuccessful) {
    pave->AddText(0.1, YY, "Minimization successful");
    YY += gNewLineSmall;
    pave->AddText(0.1, YY, Form("u_{0} = %s cm; #Delta #phi = %5.7f (rad) = %5.7f (deg)",
                                vector(stat.optimalU0),
                                stat.optimalDPhi,
                                stat.optimalDPhi * TMath::RadToDeg()
    ));
    YY += gNewLineSmall;
    pave->AddLine(0.1, YY, 0.9, YY);

    YY += gNewLineLarge;
    pave->AddText(0.1, YY, Form("Slave TPC center: %s #rightarrow %s; #Delta = %s",
                                vector(stat.slaveChamberOldCenter),
                                vector(stat.slaveChamberNewCenter),
                                vector(stat.slaveChamberNewCenter - stat.slaveChamberOldCenter)
    ));
  }

  return pave;
}

void ReadBranchesFromTree(TTree &tree) {
  tree.SetBranchAddress("slave_Y", &gTpcCalibData.slave_Y);
  tree.SetBranchAddress("slave_X", &gTpcCalibData.slave_X);
  tree.SetBranchAddress("master_Y", &gTpcCalibData.master_Y);
  tree.SetBranchAddress("master_X", &gTpcCalibData.master_X);
  tree.SetBranchAddress("slave_recVDrift", &gTpcCalibData.slave_recVDrift);

  tree.SetBranchAddress("slave_recChamberXCenter", &gTpcCalibData.slave_recChamberXCenter);
  tree.SetBranchAddress("slave_recChamberYCenter", &gTpcCalibData.slave_recChamberYCenter);

  tree.SetBranchAddress("eventUnixTime", &gTpcCalibData.eventUnixTime);
  tree.SetBranchAddress("eventNumber", &gTpcCalibData.eventNumber);
  tree.SetBranchAddress("runNumber", &gTpcCalibData.runNumber);
}

void InitOutputTree(TTree &tree, SliceAnalysisData &data) {
  tree.Branch("sliceID", &data.sliceID);
  tree.Branch("sliceEventStart", &data.sliceEventStart);
  tree.Branch("sliceEventEnd", &data.sliceEventEnd);
  tree.Branch("sliceUnixTimeStart", &data.sliceUnixTimeStart);
  tree.Branch("sliceUnixTimeEnd", &data.sliceUnixTimeEnd);

  tree.Branch("sliceMeanDUX", &data.sliceMeanDU[0]);
  tree.Branch("sliceMeanDUY", &data.sliceMeanDU[1]);

  tree.Branch("U0X", &data.optimalU0[0]);
  tree.Branch("U0Y", &data.optimalU0[1]);
  tree.Branch("A1", &data.optimalA[0][0]);
  tree.Branch("A2", &data.optimalA[0][1]);
  tree.Branch("A3", &data.optimalA[1][0]);
  tree.Branch("A4", &data.optimalA[1][1]);

  tree.Branch("slaveChamberDPhi", &data.slaveChamberDPhi);
  tree.Branch("slaveChamberDX", &data.slaveChamberDU[0]);
  tree.Branch("slaveChamberDY", &data.slaveChamberDU[1]);
}

int main(int argc, char **argv) {
  using namespace std;
  using namespace TMath;
  using namespace RooFit;

  const char *MODULE_NAME = "main";

  string input_file;
  string tree_name;

  int opt = -1;
  while (-1 != (opt = getopt(argc, argv, "i:b:"))) {
    switch (opt) {
      case 'i': input_file = string(optarg);
        break;
      case 'b': tree_name = string(optarg);
        break;
      default: break;
    }
  }

  Info(MODULE_NAME, "Input file: %s", input_file.c_str());

  auto chainPtr = make_unique<TChain>(tree_name.c_str());
  chainPtr->Add(input_file.c_str());
  chainPtr->ls();

  Info(MODULE_NAME, "Loading chain: %s", tree_name.c_str());

  ReadBranchesFromTree(*chainPtr);
  long ne = chainPtr->GetEntries();

  /*
   * Initialize output PDF
   */
  auto c = std::make_unique<TCanvas>("c1", "");
  gStyle->SetOptStat(111111);
  auto writePDF = [tree_name, &c](int option = 0) {
    const char *name = Form("alignment_%s.pdf", tree_name.c_str());
    if (option == 1) {
      name = Form("%s(", name);
    } else if (option == 2) {
      name = Form("%s)", name);
    }

    c->Print(name, "pdf");
  };

  auto sliceLabel = make_unique<TText>();
  sliceLabel->SetNDC();
  writePDF(1);

  auto outputRootFilePtr = make_unique<TFile>(Form("alignment_%s.root", tree_name.c_str()), "recreate");
  auto analysisResultTreePtr = new TTree(tree_name.c_str(), "");
  SliceAnalysisData analysisStat;
  InitOutputTree(*analysisResultTreePtr, analysisStat);

  TH2D hdYvsY_before("hdYvsY_before", ";Y (cm);dY (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdYvsY_after("hdYvsY_after", ";Y (cm);dY (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdYvsX_before("hdYvsX_before", ";X (cm);dY (cm)", 600, -300, 300, 1000, -1., 1.);
  TH2D hdYvsX_after("hdYvsX_after", ";X (cm);dY (cm)", 600, -300, 300, 1000, -1., 1.);

  TH2D hdXvsY_before("hdXvsY_before", ";Y (cm);dX (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdXvsY_after("hdXvsY_after", ";Y (cm);dX (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdXvsX_before("hdXvsX_before", ";X (cm);dX (cm)", 600, -300, 300, 1000, -1., 1.);
  TH2D hdXvsX_after("hdXvsX_after", ";X (cm);dX (cm)", 600, -300, 300, 1000, -1., 1.);

  TH1D hU0X("hU0X", ";X_{0} (cm)", 2000, -1., 1.);
  TH1D hU0Y("hU0Y", ";Y_{0} (cm)", 2000, -1., 1.);

  auto sliceData = make_shared<SliceData>();
  auto sliceDataIter = sliceData->begin();

  long nSlices = ne / N_SLICE_ENTRIES;
  long sliceStep = nSlices / N_SLICES_PRINT;

  int iSlice = 0;
  for (long ie = 0; ie < ne; ++ie) {
    chainPtr->GetEntry(ie);

    if (sliceDataIter == end(*sliceData)) {
      analysisStat.sliceID = iSlice;
      analysisStat.sliceEventStart = sliceData->front().eventNumber;
      analysisStat.sliceEventEnd = sliceData->back().eventNumber;
      analysisStat.sliceUnixTimeStart = sliceData->front().eventUnixTime;
      analysisStat.sliceUnixTimeEnd = sliceData->back().eventUnixTime;

      Info("loop", "Slice %d", iSlice);

      TVectorD meanDU(2);
      for_each(sliceData->begin(), sliceData->end(), [&meanDU](const TpcCalibData d) {
        meanDU += 1.0 / N_SLICE_ENTRIES * (d.u() - d.uPrim());
      });
      analysisStat.sliceMeanDU = meanDU;

      SliceDiscrepancyFCT fct(sliceData);

      ROOT::Math::GSLMinimizer min(ROOT::Math::kVectorBFGS);
      min.SetMaxFunctionCalls(10000);
      min.SetMaxIterations(10000);
      min.SetFunction(fct);

      min.SetVariable(0, "dPhi", 1, 1e-4);
      min.SetVariable(1, "U0X", hU0X.GetEntries() > 20 ? 0.5 * (hU0X.GetMean() + meanDU[0]) : meanDU[0], 1e-4);
      min.SetVariable(2, "U0Y", hU0Y.GetEntries() > 20 ? 0.5 * (hU0Y.GetMean() + meanDU[1]) : meanDU[1], 1e-4);

      if (true == (analysisStat.minimizationIsSuccessful = min.Minimize())) {
        Info("min", "Minimization successful");
        min.PrintResult();
        int ncalls = min.NCalls();
        int niter = min.NIterations();

        auto minX = TVectorD(fct.NDim(), min.X());
        double val = fct(minX.GetMatrixArray());

        double dPhi = minX[0];
        TVectorD u0opt(2, &minX[1]);  //  u0.Print();

        Info("min", "dPhi = %f (rad)", dPhi);

        TMatrixD optimalA(2, 2); //  A.Print();
        {
          double arr[] = {Cos(dPhi), Sin(dPhi), -Sin(dPhi), Cos(dPhi)};
          optimalA.SetMatrixArray(arr);
        }

        analysisStat.optimalA = optimalA;
        analysisStat.optimalDPhi = dPhi;
        analysisStat.optimalU0 = u0opt;

        analysisStat.slaveChamberOldCenter = sliceData->back().C();
        analysisStat.slaveChamberNewCenter = optimalA * analysisStat.slaveChamberOldCenter + u0opt;
        analysisStat.slaveChamberDU = analysisStat.slaveChamberNewCenter - analysisStat.slaveChamberOldCenter;
        analysisStat.slaveChamberDPhi = RadToDeg() * dPhi;

        analysisResultTreePtr->Fill();

        hU0X.Fill(u0opt[0]);
        hU0Y.Fill(u0opt[1]);
        /*
        TDecompSVD svd(optimalA);
        if (true == (analysisStat.svdIsSuccessful = svd.Decompose())) {
          Info("SVD", "Decomposition successful");
          auto u = svd.GetU();
          auto v = svd.GetV();
          auto sig = svd.GetSig();
          double phi_u = ASin(u[0][1]);
          double phi_v = ASin(v[0][1]);


          analysisStat.svdU = u;
          analysisStat.svdV = v;
          analysisStat.svdSig = sig;
          analysisStat.svdPhiU = phi_u;
          analysisStat.svdPhiV = phi_v;



          optimalA = u * v.T();
        }
        */

        TH2D hdYvsY_before_slice
            ("hdYvsY_before_slice", ";Y_{master} (cm);Y_{slave} - Y_{master} (cm)", 200, -100, 100, 1000, -1., 1.);
        TH2D hdYvsY_after_slice
            ("hdYvsY_after_slice", ";Y_{master} (cm);Y_{slave} - Y_{master} (cm)", 200, -100, 100, 1000, -1., 1.);
        TH2D hdYvsX_before_slice
            ("hdYvsX_before_slice", ";X_{master} (cm);Y_{slave} - Y_{master} (cm)", 600, -300, 300, 1000, -1., 1.);
        TH2D hdYvsX_after_slice
            ("hdYvsX_after_slice", ";X_{master} (cm);Y_{slave} - Y_{master} (cm)", 600, -300, 300, 1000, -1., 1.);

        for (auto data : *sliceData) {
          hdYvsY_before_slice.Fill(data.uPrim()[1], (data.u() - data.uPrim())[1]);
          hdYvsY_before.Fill(data.uPrim()[1], (data.u() - data.uPrim())[1]);
          hdYvsY_after_slice.Fill(data.uPrim()[1], (optimalA * data.u() + u0opt - data.uPrim())[1]);
          hdYvsY_after.Fill(data.uPrim()[1], (optimalA * data.u() + u0opt - data.uPrim())[1]);
          hdYvsX_before_slice.Fill(data.uPrim()[0], (data.u() - data.uPrim())[1]);
          hdYvsX_before.Fill(data.uPrim()[0], (data.u() - data.uPrim())[1]);
          hdYvsX_after_slice.Fill(data.uPrim()[0], (optimalA * data.u() + u0opt - data.uPrim())[1]);
          hdYvsX_after.Fill(data.uPrim()[0], (optimalA * data.u() + u0opt - data.uPrim())[1]);

          hdXvsY_before.Fill(data.uPrim()[1], (data.u() - data.uPrim())[0]);
          hdXvsY_after.Fill(data.uPrim()[1], (optimalA * data.u() + u0opt - data.uPrim())[0]);
          hdXvsX_before.Fill(data.uPrim()[0], (data.u() - data.uPrim())[0]);
          hdXvsX_after.Fill(data.uPrim()[0], (optimalA * data.u() + u0opt - data.uPrim())[0]);
        }

//        if (true) {
        if (sliceStep == 0 || iSlice % sliceStep == 0) {
          c->Clear();

          auto slideContent = unique_ptr<TPaveText>(GenerateSliceInfo(analysisStat));
          slideContent->Draw();
          writePDF();
          c->Clear();

          c->Divide(2, 2);
          c->cd(1);
          gPad->SetGridy();
          hdYvsY_before_slice.SetMarkerColor(kBlue);
          hdYvsY_after_slice.SetMarkerColor(kRed);
          hdYvsY_after_slice.SetLineColor(kRed);
          hdYvsY_before_slice.Draw();
          hdYvsY_after_slice.Draw("same");

          c->cd(2);
          gPad->SetGridy();
          hdYvsX_before_slice.SetMarkerColor(kBlue);
          hdYvsX_after_slice.SetMarkerColor(kRed);
          hdYvsX_after_slice.SetLineColor(kRed);
          hdYvsX_before_slice.Draw();
          hdYvsX_after_slice.Draw("same");

          c->cd(3);

          auto hdY_before = unique_ptr<TH1>(hdYvsY_before_slice.ProjectionY("hdY_before"));
          hdY_before->Draw();
          auto hdY_after = unique_ptr<TH1>(hdYvsY_after_slice.ProjectionY("hdY_after"));
          hdY_after->Draw("same");

          c->cd(4);

          writePDF();
        }
      }

      // init new slice
      iSlice++;
      sliceDataIter = begin(*sliceData);
    }

    *sliceDataIter = gTpcCalibData;
    sliceDataIter++;
  }

  c->Clear();
  c->Divide(2, 2);

  c->cd(1);
  gPad->SetGridy();
  hdYvsY_before.Draw("colz");
  c->cd(2);
  gPad->SetGridy();
  hdYvsY_after.Draw("colz");
  c->cd(3);
  gPad->SetGridy();
  hdYvsX_before.Draw("colz");
  c->cd(4);
  gPad->SetGridy();
  hdYvsX_after.Draw("colz");
  writePDF();

  c->Clear();
  c->Divide(2, 2);

  c->cd(1);
  gPad->SetGridy();
  hdXvsY_before.Draw("colz");
  c->cd(2);
  gPad->SetGridy();
  hdXvsY_after.Draw("colz");
  c->cd(3);
  gPad->SetGridy();
  hdXvsX_before.Draw("colz");
  c->cd(4);
  gPad->SetGridy();
  hdXvsX_after.Draw("colz");
  writePDF();

  c->Clear();
  c->Divide(1, 2);

  c->cd(1);
  gPad->SetLogy();
  hU0X.Draw();
  c->cd(2);
  gPad->SetLogy();
  hU0Y.Draw();

  writePDF();

  c->Clear();
  writePDF(2);

  outputRootFilePtr->cd();
  analysisResultTreePtr->Write();
  outputRootFilePtr->Close();

  return 0;
}