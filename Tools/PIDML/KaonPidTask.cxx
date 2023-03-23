// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// This task produces invariant mass vs. momentum, dEdX in TPC vs. momentum and TOF beta vs. momentum plots
/// for Kaons using ML PID from the PID ML ONNX Model and standard PID method.

#include <cmath>
#include <memory>
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Multiplicity.h"
#include "TLorentzVector.h"
#include "TDatabasePDG.h"
#include "Framework/AnalysisDataModel.h"
#include "Tools/PIDML/pidOnnxModel.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/PIDResponse.h"
#include "CommonConstants/PhysicsConstants.h"
#include "TMath.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
using MyCollisions = soa::Join<aod::Collisions,
                               aod::EvSels,
                               aod::Mults>;
using MyTracks = soa::Join<aod::FullTracks, aod::TracksExtra, aod::pidTOFbeta,
                           aod::pidTPCKa, aod::pidTOFKa, aod::TOFSignal,
			   aod::TracksDCA, aod::McTrackLabels, aod::TrackSelection>;
using MyCollision = MyCollisions::iterator;
using MyTrack = MyTracks::iterator;
} // namespace o2::aod

struct KaonPidTask {
  std::shared_ptr<PidONNXModel> pidModel; // creates a shared pointer to a new instance 'pidmodel'.
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  Configurable<float> cfgZvtxCut{"cfgZvtxCut", 10, "Z vtx cut"};
  Configurable<float> cfgEtaCut{"cfgEtaCut", 0.8, "Pseudorapidity cut"};
  Configurable<float> cfgMaxPtCut{"cfgMaxPtCut", 3.0, "Max Pt cut"};
  Configurable<float> cfgMinPtCut{"cfgMinPtCut", 0.2, "Min Pt cut"};
  Configurable<float> cfgMinNSigmaTPCCut{"cfgMinNSigmaTPCCut", 3., "N-sigma TPC cut"};
  Configurable<float> cfgChargeCut{"cfgChargeCut", 0., "N-sigma TPC cut"};
  Configurable<std::string> cfgPathLocal{"local-path", ".", "base path to the local directory with ONNX models"};
  Configurable<std::string> cfgPathCCDB{"ccdb-path", "Users/m/mkabus/PIDML", "base path to the CCDB directory with ONNX models"};
  Configurable<std::string> cfgCCDBURL{"ccdb-url", "http://alice-ccdb.cern.ch", "URL of the CCDB repository"};
  Configurable<int> cfgPid{"pid", 321, "PID to predict"};
  Configurable<double> cfgCertainty{"certainty", 0.5, "Minimum certainty above which the model accepts a particular type of particle"};
  Configurable<uint32_t> cfgDetector{"detector", kTPCTOFTRD, "What detectors to use: 0: TPC only, 1: TPC + TOF, 2: TPC + TOF + TRD"};
  Configurable<uint64_t> cfgTimestamp{"timestamp", 0, "Fixed timestamp"};
  Configurable<bool> cfgUseCCDB{"useCCDB", false, "Whether to autofetch ML model from CCDB. If false, local file will be used."};
  Configurable<bool> cfgUseMLPID{"useMLPID", true, "Whether to use ML ONNX model for PID or the standard method"};

  o2::ccdb::CcdbApi ccdbApi;

  Filter collisionFilter = (nabs(aod::collision::posZ) < cfgZvtxCut);
  Filter trackFilter = (nabs(aod::track::eta) < cfgEtaCut) && (aod::track::pt > cfgMinPtCut) && (aod::track::pt < cfgMaxPtCut);

  // Applying filters
  using MyFilteredCollisions = soa::Filtered<o2::aod::MyCollisions>;
  using MyFilteredCollision = MyFilteredCollisions::iterator;

  Partition<o2::aod::MyTracks> positive = (nabs(aod::track::eta) < cfgEtaCut) && (aod::track::pt > cfgMinPtCut) && (aod::track::pt < cfgMaxPtCut) && (aod::track::signed1Pt > cfgChargeCut);
  Partition<o2::aod::MyTracks> negative = (nabs(aod::track::eta) < cfgEtaCut) && (aod::track::pt > cfgMinPtCut) && (aod::track::pt < cfgMaxPtCut) && (aod::track::signed1Pt < cfgChargeCut);

  void init(o2::framework::InitContext&)
  {
    AxisSpec vtxZAxis = {100, -20, 20};
    std::vector<double> ptBinning = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8, 3.2, 3.6, 4.};
    AxisSpec ptAxis = {ptBinning, "#it{p}_{T} (GeV/#it{c})"};

    if (cfgUseCCDB) {
      ccdbApi.init(cfgCCDBURL); // Initializes ccdbApi when cfgUseCCDB is set to 'true'
    }
    if (cfgUseMLPID) {
      pidModel = std::make_shared<PidONNXModel>(cfgPathLocal.value, cfgPathCCDB.value, cfgUseCCDB.value, ccdbApi, cfgTimestamp.value, cfgPid.value, static_cast<PidMLDetector>(cfgDetector.value), cfgCertainty.value);
    }

    histos.add("hEta", ";#eta", kTH1F, {{100, -1.5, 1.5}});
    histos.add("hPt", ";#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0., 5.}});
    histos.add("hPhi", ";#phi (rad)", kTH1F, {{100, 0., 6.2831}});
    histos.add("hChargePos", ";z;", kTH1F, {{3, -1.5, 1.5}});
    histos.add("hChargeNeg", ";z;", kTH1F, {{3, -1.5, 1.5}});
    histos.add("hInvariantMass", ";M_{k^{+}k^{-}} (GeV/#it{c}^{2});", kTH1F, {{20, 1.0, 1.04}});
    histos.add("hdEdXvsMomentum", ";P_{K^{+}K^{-}}; dE/dx in TPC (keV/cm)", kTH2F, {{100, 0., 4.}, {200, 20., 400.}});
    histos.add("hTOFBetavsMomentum", ";P_{K^{+}K^{-}}; TOF #beta", kTH2F, {{200, 0., 5.}, {250, 0.4, 1.1}}); 
    histos.add("hPtTrueMC", ";#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0., 5.}});
    histos.add("hPtFalseMC", ";#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0., 5.}});
    histos.add("hPtTrueML", ";#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0., 5.}});
    histos.add("hPtFalseML", ";#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0., 5.}});
    histos.add("hPtFalseNeg", ";#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0., 5.}});
  }

  bool IsKaonNSigma(float mom, float nsigmaTPCK, float nsigmaTOFK)
  {
    bool fNsigmaTPCTOF = true;
    double fNsigma = 3;
    double fNsigma2 = 3;
    if (fNsigmaTPCTOF) {
      if (mom > 0.5) {
        if (mom < 2.0) {
          if (TMath::Hypot(nsigmaTOFK, nsigmaTPCK) < fNsigma)
            return true;
        } else if (TMath::Hypot(nsigmaTOFK, nsigmaTPCK) < fNsigma2)
          return true;
      } else {
        if (TMath::Abs(nsigmaTPCK) < fNsigma)
          return true;
      }
    } else {

      if (mom < 0.4) {
        if (nsigmaTOFK < -999.) {
          if (TMath::Abs(nsigmaTPCK) < 2.0)
            return true;
        } else if (TMath::Abs(nsigmaTOFK) < 3.0 && TMath::Abs(nsigmaTPCK) < 3.0)
          return true;
      } else if (mom >= 0.4 && mom <= 0.6) {
        if (nsigmaTOFK < -999.) {
          if (TMath::Abs(nsigmaTPCK) < 2.0)
            return true;
        } else if (TMath::Abs(nsigmaTOFK) < 3.0 && TMath::Abs(nsigmaTPCK) < 3.0)
          return true;
      } else if (nsigmaTOFK < -999.) {
        return false;
      } else if (TMath::Abs(nsigmaTOFK) < 3.0 && TMath::Abs(nsigmaTPCK) < 3.0)
        return true;
    }
    return false;
 }

  void process(MyFilteredCollision const& coll, o2::aod::MyTracks const& tracks, o2::aod::McParticles const& mctracks)
  {
    auto groupPositive = positive->sliceByCached(aod::track::collisionId, coll.globalIndex());
    auto groupNegative = negative->sliceByCached(aod::track::collisionId, coll.globalIndex());
    
    for (const auto& track : groupPositive) {
      if (!track.has_mcParticle()) {
       continue;
      }
      const auto mcParticle = track.mcParticle_as<aod::McParticles>();
       if (mcParticle.pdgCode() == cfgPid.value) {  //condition for true MC
       	histos.fill(HIST("hPtTrueMC"), track.pt());
       } 
        else { 
       	 histos.fill(HIST("hPtFalseMC"), track.pt());  //condition for false MC
       }

      histos.fill(HIST("hChargePos"), track.sign());
      histos.fill(HIST("hEta"), track.eta());
      histos.fill(HIST("hPt"), track.pt());
      histos.fill(HIST("hPhi"), track.phi());
      
      if ((cfgUseMLPID.value && pidModel.get()->applyModelBoolean(track)) ||
      (!cfgUseMLPID.value && IsKaonNSigma(track.p(), track.tpcNSigmaKa(), track.tofNSigmaKa()))) {
	if (mcParticle.pdgCode() == cfgPid.value) {
          histos.fill(HIST("hPtTrueML"), track.pt());  //condition for true positives
	} 
	else { 
          histos.fill(HIST("hPtFalseML"), track.pt());  //condition for false positives
        }

	histos.fill(HIST("hdEdXvsMomentum"), track.p(), track.tpcSignal());
        histos.fill(HIST("hTOFBetavsMomentum"), track.p(), track.beta());
      }
       else { 
	if (mcParticle.pdgCode() == cfgPid.value) {
        histos.fill(HIST("hPtFalseNeg"), track.pt());  //condition for false negatives
       }
     } 
   }

    for (auto track : groupNegative) {
      if (!track.has_mcParticle()) {
       continue;
      }
      const auto mcParticle = track.mcParticle_as<aod::McParticles>();
       if (mcParticle.pdgCode() == cfgPid.value) {  //condition for true MC
       	histos.fill(HIST("hPtTrueMC"), track.pt());
       } 
        else { 
       	 histos.fill(HIST("hPtFalseMC"), track.pt());  //condition for false MC
       }

      histos.fill(HIST("hChargeNeg"), track.sign());
      histos.fill(HIST("hEta"), track.eta());
      histos.fill(HIST("hPt"), track.pt());
      histos.fill(HIST("hPhi"), track.phi());

      if ((cfgUseMLPID.value && pidModel.get()->applyModelBoolean(track)) ||
      (!cfgUseMLPID.value && IsKaonNSigma(track.p(), track.tpcNSigmaKa(), track.tofNSigmaKa()))) {
	if (mcParticle.pdgCode() == cfgPid.value) {
          histos.fill(HIST("hPtTrueML"), track.pt());  //condition for true positives
	} 
	else { 
          histos.fill(HIST("hPtFalseML"), track.pt());  //condition for false positives
        }

        histos.fill(HIST("hdEdXvsMomentum"), track.p(), track.tpcSignal());
        histos.fill(HIST("hTOFBetavsMomentum"), track.p(), track.beta());
      }
       else { 
	if (mcParticle.pdgCode() == cfgPid.value) {
        histos.fill(HIST("hPtFalseNeg"), track.pt());  //condition for false negatives
    }
  }
}

    for (auto& [pos, neg] : combinations(soa::CombinationsFullIndexPolicy(groupPositive, groupNegative))) {
      if (cfgUseMLPID.value && (!(pidModel.get()->applyModelBoolean(pos)) || !(pidModel.get()->applyModelBoolean(neg)))) {
        continue;
      }
      if (!cfgUseMLPID.value && (!(IsKaonNSigma(pos.p(), pos.tpcNSigmaKa(), pos.tofNSigmaKa())) || !(IsKaonNSigma(neg.p(), neg.tpcNSigmaKa(), neg.tofNSigmaKa())))) {
	continue;
      }

      TLorentzVector part1Vec;
      TLorentzVector part2Vec;
      float mMassOne = TDatabasePDG::Instance()->GetParticle(cfgPid.value)->Mass();
      float mMassTwo = TDatabasePDG::Instance()->GetParticle(cfgPid.value)->Mass();

      part1Vec.SetPtEtaPhiM(pos.pt(), pos.eta(), pos.phi(), mMassOne);
      part2Vec.SetPtEtaPhiM(neg.pt(), neg.eta(), neg.phi(), mMassTwo);

      TLorentzVector sumVec(part1Vec);
      sumVec += part2Vec;

      histos.fill(HIST("hInvariantMass"), sumVec.M());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<KaonPidTask>(cfgc)};
  return workflow;
}
