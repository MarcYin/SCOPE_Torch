from pathlib import Path

import torch

from scope.spectral.fluspect import FluspectModel, LeafBioBatch
from scope.spectral.loaders import load_fluspect_resources, load_scope_filenames, load_soil_spectra, scope_repo_root


def test_load_scope_filenames_reads_default_assets():
    filenames = load_scope_filenames()

    assert filenames["optipar_file"] == "Optipar2021_ProspectPRO_CX.mat"
    assert filenames["soil_file"] == "soilnew.txt"


def test_load_fluspect_resources_reads_default_optipar():
    resources = load_fluspect_resources(dtype=torch.float64)

    assert (
        resources.source
        == scope_repo_root() / "upstream" / "SCOPE" / "input" / "fluspect_parameters" / "Optipar2021_ProspectPRO_CX.mat"
    )
    assert resources.spectral.wlP.shape == (2001,)
    assert float(resources.spectral.wlP[0]) == 400.0
    assert float(resources.spectral.wlP[-1]) == 2400.0
    assert resources.optipar.Kp is not None
    assert resources.optipar.Kcbc is not None
    assert resources.extras["GSV"].shape == (2001, 3)
    assert resources.extras["phiE"].shape == (2001,)


def test_load_fluspect_resources_handles_legacy_optipar():
    resources = load_fluspect_resources("Optipar2017_ProspectD.mat")

    assert (
        resources.source
        == scope_repo_root() / "upstream" / "SCOPE" / "input" / "fluspect_parameters" / "Optipar2017_ProspectD.mat"
    )
    assert resources.optipar.Kp is None
    assert resources.optipar.Kcbc is None
    assert "phiE" not in resources.extras


def test_load_soil_spectra_reads_soil_library():
    soils = load_soil_spectra(dtype=torch.float64)

    assert soils.source == scope_repo_root() / "upstream" / "SCOPE" / "input" / "soil_spectra" / "soilnew.txt"
    assert soils.wavelength.shape == (2001,)
    assert soils.spectra.shape == (2001, 3)
    assert soils.names == ("soil1", "soil2", "soil3")
    assert torch.allclose(soils.spectrum("soil2"), soils.spectra[:, 1])


def test_loaders_accept_explicit_repo_relative_paths():
    repo_root = scope_repo_root()
    resources = load_fluspect_resources(
        repo_root / "upstream" / "SCOPE" / "input" / "fluspect_parameters" / "Optipar2021_ProspectPRO_CX.mat"
    )
    soils = load_soil_spectra(Path("upstream/SCOPE/input/soil_spectra/soilnew.txt"))

    assert resources.optipar.Kp is not None
    assert soils.spectra.shape[1] == 3


def test_fluspect_model_from_scope_assets_runs():
    model = FluspectModel.from_scope_assets(dtype=torch.float64)
    leafbio = LeafBioBatch(Cab=torch.tensor([40.0], dtype=torch.float64))
    output = model(leafbio)

    assert output.refl.shape == (1, 2001)
    assert output.tran.shape == (1, 2001)
