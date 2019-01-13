//
// Created by eugene on 1/13/19.
//

#ifndef NEWTPCCALIBRATION_TYPES_H
#define NEWTPCCALIBRATION_TYPES_H

#include <Rtypes.h>
#include <TVectorD.h>
#include <TMatrixD.h>

struct TpcCalibData {
  uint eventUnixTime;
  uint eventNumber;
  uint runNumber;

  double slave_Y;
  double slave_X;
  double master_Y;
  double master_X;
  double slave_recVDrift;

  double slave_recChamberXCenter;
  double slave_recChamberYCenter;

  TVectorD u() const {
    double uArr[2] = {slave_X, slave_Y};
    return TVectorD(2, uArr);
  }

  TVectorD uPrim() const {
    double uArr[2] = {master_X, master_Y};
    return TVectorD(2, uArr);
  }

  TVectorD C() const {
    double xcArr[] = {slave_recChamberXCenter, slave_recChamberYCenter};
    return TVectorD(2, xcArr);
  }

};

class SliceAnalysisData : public TObject {

 public:
  int sliceID{-1};

  uint sliceEventStart{0};
  uint sliceEventEnd{0};
  uint sliceUnixTimeStart{0};
  uint sliceUnixTimeEnd{0};

  bool isMinimizationSuccessful{false};
  TMatrixD Aopt{TMatrixD(2, 2)};
  TVectorD u0opt{TVectorD(2)};

  bool isSVDSuccessful{false};
  TMatrixD svdU{TMatrixD(2, 2)}, svdV{TMatrixD(2, 2)};
  TVectorD signal{TVectorD(2)};
  double svdPhiU{-999}, svdPhiV{-999};

  TVectorD oldC{TVectorD(2)}, newC{TVectorD(2)};

  ClassDef (SliceAnalysisData,1)

};

#endif //NEWTPCCALIBRATION_TYPES_H
