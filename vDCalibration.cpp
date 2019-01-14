//
// Created by eugene on 1/14/19.
//

#include <iostream>
#include <string>
#include <TChain.h>
#include <TNtupleD.h>
#include <TError.h>
#include <TCanvas.h>

int main(int argc, char **argv) {
  using namespace std;

  string tpc_name = "VTPC1";
  string tree_name = "VTPC1vsVTPC2";
  string input_root = "alignment_" + tree_name + ".root";
  string input_txt = "../input/VTPC1.txt";

  TNtupleD recVDriftTree("recVDriftTree", "", "time:vD");
  recVDriftTree.ReadFile(input_txt.c_str());

  double recVDrift;
  double time;
  recVDriftTree.SetBranchAddress("vD", &recVDrift);
  recVDriftTree.SetBranchAddress("time", &time);

  TNtupleD calibratedVDriftTree("calibratedVDriftTree", "", "time:vD");

  TChain analysisChain(tree_name.c_str());
  analysisChain.Add(input_root.c_str());

  uint sliceUnixTimeStart, sliceUnixTimeEnd;
  double dYYSlope, dYYSlopeError;
  double sliceRecVDrift;

  analysisChain.SetBranchAddress("sliceUnixTimeStart", &sliceUnixTimeStart);
  analysisChain.SetBranchAddress("sliceUnixTimeEnd", &sliceUnixTimeEnd);
  analysisChain.SetBranchAddress("dYYSlope", &dYYSlope);
  analysisChain.SetBranchAddress("dYYSlopeError", &dYYSlopeError);
  analysisChain.SetBranchAddress("sliceRecVDrift", &sliceRecVDrift);

  uint calibrationTimeStart;
  uint calibrationTimeEnd;
  analysisChain.GetEntry(0);
  calibrationTimeStart = sliceUnixTimeStart;
  analysisChain.GetEntry(analysisChain.GetEntriesFast());
  calibrationTimeEnd = sliceUnixTimeEnd;
  Info("vDCalibration", "Interval [%u : %u] s", calibrationTimeStart, calibrationTimeEnd);

  for (long ie = 0; ie < analysisChain.GetEntriesFast(); ++ie) {
    analysisChain.GetEntry(ie);
    double calibTime = (sliceUnixTimeEnd + sliceUnixTimeStart) / 2.0;
    double dt = sliceUnixTimeEnd - sliceUnixTimeStart;

    if (dt > 600) {
      Warning("vDCalibration", "dt > 600. Skipping...");
      continue;
    }

    double calibratedVDrift = sliceRecVDrift * (1. / (1 - dYYSlope));
    double calibratedVDriftError = calibratedVDrift * dYYSlopeError / sliceRecVDrift;
    Info("vDCalibration", "T = %f vD = %f eps = +- %f ", calibTime, calibratedVDrift, calibratedVDriftError);
    calibratedVDriftTree.Fill(calibTime, calibratedVDrift);
  }

  auto c = new TCanvas;
  calibratedVDriftTree.Draw("vD:time>>pCalibVDriftVsTime", "", "prof");
  recVDriftTree.Draw("vD:time>>pRecVDriftVsTime", "", "prof,same");

  c->Print("vD_QA.pdf", "pdf");


  return 0;
}