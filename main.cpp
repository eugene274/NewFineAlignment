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
#include <TH2D.h>

constexpr int N_PAR_DIMENSION = 6;
constexpr int N_SLICE_ENTRIES = 5000;
constexpr int N_SLICES_PRINT = 10;

struct TpcCalibData {
  uint eventUnixTime;
  uint eventNumber;
  uint runNumber;

  double slave_Y;
  double slave_X;
  double master_Y;
  double master_X;
  double slave_recVDrift;

  TVectorD u() const {
    double uArr[2] = {slave_X, slave_Y};
    return TVectorD(2, uArr);
  }

  TVectorD uPrim() const {
    double uArr[2] = {master_X, master_Y};
    return TVectorD(2, uArr);
  }

} gTpcCalibData;

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

void ReadBranchesFromTree(TTree &tree) {
  tree.SetBranchAddress("slave_Y", &gTpcCalibData.slave_Y);
  tree.SetBranchAddress("slave_X", &gTpcCalibData.slave_X);
  tree.SetBranchAddress("master_Y", &gTpcCalibData.master_Y);
  tree.SetBranchAddress("master_X", &gTpcCalibData.master_X);
  tree.SetBranchAddress("slave_recVDrift", &gTpcCalibData.slave_recVDrift);

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

  auto ifile = unique_ptr<TFile>(TFile::Open(input_file.c_str(), "read"));

  if (!ifile) {
    return 1;
  }

  Info(MODULE_NAME, "Input file: %s", input_file.c_str());

  auto tree = unique_ptr<TTree>((TTree *) ifile->Get(tree_name.c_str()));

  /*
   * Initialize tree
   */
  if (!tree) {
    return 1;
  }

  Info(MODULE_NAME, "Loading tree: %s", tree_name.c_str());

  ReadBranchesFromTree(*tree);
  long ne = tree->GetEntries();

  /*
   * Initialize output
   */
  auto c = std::make_unique<TCanvas>("c1", "");
  gStyle->SetOptStat(111111);

  auto sliceLabel = make_unique<TText>();
  sliceLabel->SetNDC();

  c->Print("test.pdf(", "pdf");

  TH2D hdYvsY_before("hdYvsY_before", ";Y (cm);dY (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdYvsY_after("hdYvsY_after", ";Y (cm);dY (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdYvsX_before("hdYvsX_before", ";X (cm);dY (cm)", 600, -300, 300, 1000, -1., 1.);
  TH2D hdYvsX_after("hdYvsX_after", ";X (cm);dY (cm)", 600, -300, 300, 1000, -1., 1.);

  TH2D hdXvsY_before("hdXvsY_before", ";Y (cm);dX (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdXvsY_after("hdXvsY_after", ";Y (cm);dX (cm)", 200, -100, 100, 1000, -1., 1.);
  TH2D hdXvsX_before("hdXvsX_before", ";X (cm);dX (cm)", 600, -300, 300, 1000, -1., 1.);
  TH2D hdXvsX_after("hdXvsX_after", ";X (cm);dX (cm)", 600, -300, 300, 1000, -1., 1.);

  auto sliceData = make_shared<SliceData>();
  auto sliceDataIter = sliceData->begin();

  long nSlices = ne / N_SLICE_ENTRIES;
  long sliceStep = nSlices / N_SLICES_PRINT;

  int iSlice = 0;
  for (long ie = 0; ie < ne; ++ie) {
    tree->GetEntry(ie);

    if (sliceDataIter == end(*sliceData)) {
      Info("loop", "Slice %d", iSlice);

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
      min.SetVariable(4, "U0X", 0, 0.001);
      min.SetVariable(5, "U0Y", 0, 0.001);

      if (min.Minimize()) {
        min.PrintResult();
        int ncalls = min.NCalls();
        int niter = min.NIterations();

        auto minX = TVectorD(N_PAR_DIMENSION, min.X());
        double val = fct(minX.GetMatrixArray());

        TMatrixD Aopt(2, 2, &minX[0]); //  A.Print();
        TVectorD u0opt(2, &minX[4]);  //  u0.Print();


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

        if (iSlice % sliceStep == 0) {
          c->Clear();
          sliceLabel->DrawText(0.1, 0.8,
                               Form("Slice %d: Events %d - %d",
                                    iSlice,
                                    sliceData->front().eventNumber,
                                    sliceData->back().eventNumber));
          c->Print("test.pdf", "pdf");
          c->Clear();

          c->Divide(2, 2);
          c->cd(1);
          hdYvsY_after_slice.SetMarkerColor(kRed);
          hdYvsY_after_slice.SetLineColor(kRed);
          hdYvsY_before_slice.Draw();
          hdYvsY_after_slice.Draw("same");

          c->cd(2);

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

          auto hdX_before = unique_ptr<TH1>(hdYvsY_before_slice.ProjectionY("hdX_before"));
          hdX_before->Draw();
          auto hdX_after = unique_ptr<TH1>(hdYvsY_after_slice.ProjectionY("hdX_after"));
          hdX_after->Draw("same");

          c->Print("test.pdf", "pdf");
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
  hdYvsY_before.Draw("colz");
  c->cd(2);
  hdYvsY_after.Draw("colz");
  c->cd(3);
  hdYvsX_before.Draw("colz");
  c->cd(4);
  hdYvsX_after.Draw("colz");
  c->Print("test.pdf", "pdf");

  c->Clear();
  c->Divide(2, 2);

  c->cd(1);
  hdXvsY_before.Draw("colz");
  c->cd(2);
  hdXvsY_after.Draw("colz");
  c->cd(3);
  hdXvsX_before.Draw("colz");
  c->cd(4);
  hdXvsX_after.Draw("colz");
  c->Print("test.pdf", "pdf");

  c->Clear();
  c->Print("test.pdf)", "pdf");

  return 0;
}