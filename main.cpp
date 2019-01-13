#include <utility>

#include <utility>

#include <memory>

#include <iostream>
#include <getopt.h>
#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooPolynomial.h>
#include <RooDataSet.h>
#include <RooFormulaVar.h>
#include <RooPlot.h>
#include <TCanvas.h>
#include <TText.h>
#include <TStyle.h>
#include <TVector3.h>
#include <RooGenericPdf.h>
#include <RooGaussian.h>
#include <TVectorT.h>
#include <TMatrixDEigen.h>
#include <Math/IFunction.h>
#include <Math/GSLMinimizer.h>
#include <TDecompSVD.h>
#include <TPaveText.h>
#include <TH2D.h>
#include <TChain.h>

#include "types.h"

constexpr int N_PAR_DIMENSION = 6;
constexpr int N_SLICE_ENTRIES = 5000;
constexpr int N_SLICES_PRINT = 10;

TpcCalibData gTpcCalibData;
typedef std::array<TpcCalibData, N_SLICE_ENTRIES> SliceData;

class SliceDiscrepancyFCT : public ROOT::Math::IGradientFunctionMultiDim {

 public:
  explicit SliceDiscrepancyFCT(std::shared_ptr<SliceData> _sliceDataPtr) : sliceDataPtr(std::move(_sliceDataPtr)) {}

  unsigned int NDim() const override {
    return N_PAR_DIMENSION;
  }

  void FdF(const double *x, double &f, double *df) const override {
    using namespace std;
    TMatrixD A(2, 2, &x[0]);
    TVectorD u0(2, &x[4]);

    TVectorD meanGrad(N_PAR_DIMENSION);
    double dd = 0;
    for_each(sliceDataPtr->begin(), sliceDataPtr->end(), [A, u0, &dd, &meanGrad](const TpcCalibData &data) {
      if ((data.u() - data.uPrim()).Norm1() > 1.)
        return;

      auto delta = (A * data.u() + u0) - data.uPrim();
      dd += delta.Norm2Sqr();

      TVectorD grad(N_PAR_DIMENSION);
      grad[0] = delta[0] * data.u()[0];
      grad[1] = delta[0] * data.u()[1];
      grad[2] = delta[1] * data.u()[0];
      grad[3] = delta[1] * data.u()[1];
      grad[4] = delta[0];
      grad[5] = delta[1];
      grad *= 2.;

      meanGrad += grad;

    });

    dd /= N_SLICE_ENTRIES;
    meanGrad *= 1. / N_SLICE_ENTRIES;

    f = dd;
    double *dfArr = meanGrad.GetMatrixArray();
    copy(&dfArr[0], &dfArr[N_PAR_DIMENSION], &df[0]);
  }

  IBaseFunctionMultiDimTempl<double> *Clone() const override {
    return new SliceDiscrepancyFCT(this->sliceDataPtr);
  }

 private:
  double DoEval(const double *x) const override {
    double f;
    double fdf[6];
    FdF(x, f, fdf);
    return f;
  }

  double DoDerivative(const double *x, unsigned int icoord) const override {
    double f;
    double fdf[6];
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
  pave->AddText(0.1,
                0.8,
                Form("Slice %d: Events %u - %u",
                     stat.sliceID,
                     stat.sliceEventStart,
                     stat.sliceEventEnd))->SetTextSize(0.04);
  pave->AddText(0.1, 0.75, Form("#Delta T = %u s", stat.sliceUnixTimeEnd - stat.sliceUnixTimeStart));
  pave->AddLine(0.1, 0.7, 0.9, 0.7);
  if (stat.isMinimizationSuccessful) {
    pave->AddText(0.1, 0.68, "Minimization successful");
    pave->AddText(0.1, 0.65, Form("u_{0} = %s cm; A = %s", vector(stat.u0opt), matrix(stat.Aopt)));
    pave->AddLine(0.1, 0.58, 0.9, 0.58);
  }

  TMatrixD v = stat.svdV;
  v.T();

  if (stat.isSVDSuccessful) {
    pave->AddText(0.1, 0.53, "SVD decomposition successful");
    pave->AddText(0.1, 0.50, Form("A = U #times S #times #bar{V} = %s #times %s #times %s",
                                  matrix(stat.svdU), "S", matrix(v)));
    pave->AddText(0.1, 0.42, Form("d#phi = #phi_{U} - #phi_{V} = %5.5f - %5.5f = %5.5f (rad) = %5.5f (deg)",
                                  stat.svdPhiU,
                                  stat.svdPhiV,
                                  stat.svdPhiU - stat.svdPhiV,
                                  TMath::RadToDeg() * (stat.svdPhiU - stat.svdPhiV)));

    pave->AddLine(0.1, 0.37, 0.9, 0.37);
    pave->AddText(0.1, 0.32, Form("Slave TPC center: %s #rightarrow %s; #Delta = %s",
                                  vector(stat.oldC),
                                  vector(stat.newC),
                                  vector(stat.newC - stat.oldC)
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
  auto analysisResultTreePtr = new TTree("analysisResultTree", "");
  SliceAnalysisData analysisStat;
  analysisResultTreePtr->Branch(tree_name.c_str(), &analysisStat);

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
  for (long ie = 0; ie < 20000; ++ie) {
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

      SliceDiscrepancyFCT fct(sliceData);

      ROOT::Math::GSLMinimizer min(ROOT::Math::kVectorBFGS);
      min.SetMaxFunctionCalls(10000);
      min.SetMaxIterations(10000);
//      min.SetTolerance(0.005 * N_SLICE_ENTRIES);
      min.SetFunction(fct);

      min.SetVariable(0, "A0", 1, 0.01);
      min.SetVariable(1, "A1", 0, 0.01);
      min.SetVariable(2, "A2", 0, 0.01);
      min.SetVariable(3, "A3", 1, 0.01);
      min.SetVariable(4, "U0X", hU0X.GetEntries() > 20 ? 0.5 * (hU0X.GetMean() + meanDU[0]) : meanDU[0], 1e-4);
      min.SetVariable(5, "U0Y", hU0Y.GetEntries() > 20 ? 0.5 * (hU0Y.GetMean() + meanDU[1]) : meanDU[1], 1e-4);

      if (true == (analysisStat.isMinimizationSuccessful = min.Minimize())) {
        Info("min", "Minimization successful");
        min.PrintResult();
        int ncalls = min.NCalls();
        int niter = min.NIterations();

        auto minX = TVectorD(N_PAR_DIMENSION, min.X());
        double val = fct(minX.GetMatrixArray());

        TMatrixD Aopt(2, 2, &minX[0]); //  A.Print();
        TVectorD u0opt(2, &minX[4]);  //  u0.Print();

        analysisStat.Aopt = Aopt;
        analysisStat.u0opt = u0opt;

        hU0X.Fill(u0opt[0]);
        hU0Y.Fill(u0opt[1]);

        TDecompSVD svd(Aopt);
        if (true == (analysisStat.isSVDSuccessful = svd.Decompose())) {
          Info("decomp", "Decomposition successful");
          auto u = svd.GetU();
          auto v = svd.GetV();
          auto sig = svd.GetSig();
          double phi_u = ASin(u[0][1]);
          double phi_v = ASin(v[0][1]);
          Info("decomp", "dPhi = %f", phi_u - phi_v);

          analysisStat.svdU = u;
          analysisStat.svdV = v;
          analysisStat.signal = sig;
          analysisStat.svdPhiU = phi_u;
          analysisStat.svdPhiV = phi_v;

          analysisStat.oldC = sliceData->back().C();
          analysisStat.newC = Aopt * analysisStat.oldC + u0opt;

          analysisResultTreePtr->Fill();
        }

        TH2D hdYvsY_before_slice("hdYvsY_before_slice", ";Y (cm);dY (cm)", 200, -100, 100, 1000, -1., 1.);
        TH2D hdYvsY_after_slice("hdYvsY_after_slice", ";Y (cm);dY (cm)", 200, -100, 100, 1000, -1., 1.);
        TH2D hdYvsX_before_slice("hdYvsX_before_slice", ";X (cm);dY (cm)", 600, -300, 300, 1000, -1., 1.);
        TH2D hdYvsX_after_slice("hdYvsX_after_slice", ";X (cm);dY (cm)", 600, -300, 300, 1000, -1., 1.);

        for (auto data : *sliceData) {
          hdYvsY_before_slice.Fill(data.uPrim()[1], (data.u() - data.uPrim())[1]);
          hdYvsY_before.Fill(data.uPrim()[1], (data.u() - data.uPrim())[1]);
          hdYvsY_after_slice.Fill(data.uPrim()[1], (Aopt * data.u() + u0opt - data.uPrim())[1]);
          hdYvsY_after.Fill(data.uPrim()[1], (Aopt * data.u() + u0opt - data.uPrim())[1]);
          hdYvsX_before_slice.Fill(data.uPrim()[0], (data.u() - data.uPrim())[1]);
          hdYvsX_before.Fill(data.uPrim()[0], (data.u() - data.uPrim())[1]);
          hdYvsX_after_slice.Fill(data.uPrim()[0], (Aopt * data.u() + u0opt - data.uPrim())[1]);
          hdYvsX_after.Fill(data.uPrim()[0], (Aopt * data.u() + u0opt - data.uPrim())[1]);

          hdXvsY_before.Fill(data.uPrim()[1], (data.u() - data.uPrim())[0]);
          hdXvsY_after.Fill(data.uPrim()[1], (Aopt * data.u() + u0opt - data.uPrim())[0]);
          hdXvsX_before.Fill(data.uPrim()[0], (data.u() - data.uPrim())[0]);
          hdXvsX_after.Fill(data.uPrim()[0], (Aopt * data.u() + u0opt - data.uPrim())[0]);
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