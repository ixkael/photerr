"""Custom photometric error model with per-object parameters."""

from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from photerr.model import ErrorModel
from photerr.params import ErrorParams, param_docstring


@dataclass
class CustomErrorParams(ErrorParams):
    """Parameters for a custom photometric error model.
    
    This is a template for creating custom error models. You can modify the
    default values below to match your specific survey requirements.
    """

    __doc__ += param_docstring

    nYrObs: float = 5.0
    nVisYr: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 10.0,
            "r": 15.0,
            "i": 15.0,
            "z": 10.0,
        }
    )
    gamma: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 0.04,
            "r": 0.04,
            "i": 0.04,
            "z": 0.04,
        }
    )

    m5: dict[str, float] | float = field(default_factory=lambda: {})

    tvis: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 60,
            "r": 60,
            "i": 60,
            "z": 60,
        }
    )
    airmass: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 1.2,
            "r": 1.2,
            "i": 1.2,
            "z": 1.2,
        }
    )
    Cm: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 24.0,
            "r": 24.5,
            "i": 24.3,
            "z": 23.8,
        }
    )
    dCmInf: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 0.1,
            "r": 0.05,
            "i": 0.03,
            "z": 0.02,
        }
    )
    msky: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 22.0,
            "r": 21.0,
            "i": 20.0,
            "z": 19.0,
        }
    )
    mskyDark: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 22.5,
            "r": 21.5,
            "i": 20.5,
            "z": 19.5,
        }
    )
    theta: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 1.0,
            "r": 0.9,
            "i": 0.8,
            "z": 0.8,
        }
    )
    km: dict[str, float] | float = field(
        default_factory=lambda: {
            "g": 0.15,
            "r": 0.10,
            "i": 0.08,
            "z": 0.06,
        }
    )
    tvisRef: float = 30.0

    sigmaSys: float = 0.005
    sigLim: float = 0
    ndMode: str = "flag"
    ndFlag: float = float("inf")
    absFlux: bool = False
    extendedType: str = "point"
    aMin: float = 2.0
    aMax: float = 0.7
    majorCol: str = "major"
    minorCol: str = "minor"
    decorrelate: bool = True
    highSNR: bool = False
    errLoc: str = "after"
    scale: dict[str, float] | float = field(default_factory=lambda: {})
    
    renameDict: InitVar[dict[str, str] | None] = None
    validate: InitVar[bool] = True


class CustomErrorModel(ErrorModel):
    """Custom photometric error model with per-object parameter support.
    
    This error model extends the base ErrorModel to accept per-object parameters
    from the input DataFrame. This allows for more realistic error modeling where
    observing conditions vary between objects.
    
    Supported per-object parameters:
    - {band}_err: Direct magnitude error for each band
    - {band}_psf: PSF FWHM in arcseconds for each band
    - {band}_airmass: Airmass for each band
    - {band}_msky: Sky brightness for each band
    - {band}_tvis: Exposure time for each band
    - {band}_nvis: Number of visits for each band
    - {band}_scale: Error scaling factor for each band
    
    Example usage:
    
    ```python
    from photerr import CustomErrorModel
    import pandas as pd
    
    # Create a catalog with per-object parameters
    catalog = pd.DataFrame({
        'g': [20.0, 21.0, 22.0],
        'r': [19.5, 20.5, 21.5],
        'g_psf': [0.8, 1.0, 1.2],  # PSF varies per object
        'r_psf': [0.7, 0.9, 1.1],
        'g_airmass': [1.1, 1.2, 1.3],  # Airmass varies per object
        'r_airmass': [1.0, 1.1, 1.2]
    })
    
    # Use the custom error model
    err_model = CustomErrorModel()
    catalog_with_errors = err_model(catalog, random_state=42)
    ```
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the custom error model.
        
        Parameters are passed to the parent ErrorModel class.
        If no parameters are provided, CustomErrorParams defaults are used.
        """
        if len(args) == 0 and len(kwargs) == 0:
            # Use default CustomErrorParams
            super().__init__(CustomErrorParams())
        else:
            # Pass arguments to parent class
            super().__init__(*args, **kwargs)

    def _get_per_object_params(self, catalog: pd.DataFrame, param_name: str, bands: list) -> dict:
        """Extract per-object parameters from catalog if available.
        
        Parameters
        ----------
        catalog : pd.DataFrame
            Input catalog
        param_name : str
            Parameter name (e.g., 'psf', 'airmass', 'msky', etc.)
        bands : list
            List of bands to check for
            
        Returns
        -------
        dict
            Dictionary with band keys and numpy arrays of per-object values
        """
        per_obj_params = {}
        
        for band in bands:
            col_name = f"{band}_{param_name}"
            if col_name in catalog.columns:
                per_obj_params[band] = catalog[col_name].to_numpy()
        
        return per_obj_params

    def _get_nsr_from_mags(
        self,
        mags: np.ndarray,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
        catalog: pd.DataFrame = None,
    ) -> np.ndarray:
        """Calculate the noise-to-signal ratio with per-object parameter support.
        
        Extended version that checks for per-object parameters in the catalog.
        """
        # Check if direct magnitude errors are provided
        if catalog is not None:
            direct_errors = self._get_per_object_params(catalog, "err", bands)
            if direct_errors:
                # If direct errors are provided, use them directly
                nsr_values = []
                for i, band in enumerate(bands):
                    if band in direct_errors:
                        if self.params.highSNR:
                            nsr = direct_errors[band]
                        else:
                            nsr = 10 ** (direct_errors[band] / 2.5) - 1
                        nsr_values.append(nsr)
                    else:
                        # Fall back to calculation for this band
                        nsr = self._calculate_nsr_for_band(mags[:, i], majors, minors, band, catalog, i)
                        nsr_values.append(nsr)
                return np.column_stack(nsr_values)
        
        # Get per-object parameters from catalog if available
        per_obj_psf = self._get_per_object_params(catalog, "psf", bands) if catalog is not None else {}
        per_obj_airmass = self._get_per_object_params(catalog, "airmass", bands) if catalog is not None else {}
        per_obj_msky = self._get_per_object_params(catalog, "msky", bands) if catalog is not None else {}
        per_obj_tvis = self._get_per_object_params(catalog, "tvis", bands) if catalog is not None else {}
        per_obj_nvis = self._get_per_object_params(catalog, "nvis", bands) if catalog is not None else {}
        per_obj_scale = self._get_per_object_params(catalog, "scale", bands) if catalog is not None else {}
        
        # Build arrays for each parameter, using per-object values when available
        n_objects = mags.shape[0]
        
        # Get base parameter values
        base_gamma = np.array([self.params.gamma[band] for band in bands])
        base_nVisYr = np.array([self.params.nVisYr[band] for band in bands])
        base_scale = np.array([self.params.scale.get(band, 1.0) for band in bands])
        
        # Create arrays that can vary per object
        gamma = np.broadcast_to(base_gamma, (n_objects, len(bands)))
        nVisYr = np.broadcast_to(base_nVisYr, (n_objects, len(bands)))
        scale = np.broadcast_to(base_scale, (n_objects, len(bands)))
        
        # Override with per-object values where available
        for i, band in enumerate(bands):
            if band in per_obj_nvis:
                nVisYr[:, i] = per_obj_nvis[band]
            if band in per_obj_scale:
                scale[:, i] = per_obj_scale[band]
        
        # Calculate m5 per object (may vary due to PSF, airmass, etc.)
        m5_per_obj = self._calculate_m5_per_object(
            bands, n_objects, per_obj_psf, per_obj_airmass, per_obj_msky, per_obj_tvis
        )
        
        # Calculate x as defined in the paper
        x = 10 ** (0.4 * (mags - m5_per_obj))
        
        # Calculate the NSR for a single visit
        with np.errstate(invalid="ignore"):
            nsrRandSingleExp = np.sqrt((0.04 - gamma) * x + gamma * x**2)
        
        # Calculate the NSR for the stacked image
        nStackedObs = nVisYr * self.params.nYrObs
        nsrRand = nsrRandSingleExp / np.sqrt(nStackedObs)
        
        # Rescale according to the area ratio
        if self.params.extendedType == "auto":
            A_ratio = self._get_area_ratio_auto(majors, minors, bands)
        elif self.params.extendedType == "gaap":
            A_ratio = self._get_area_ratio_gaap(majors, minors, bands)
        else:
            A_ratio = 1
        nsrRand *= np.sqrt(A_ratio)
        
        # Apply per-object scaling
        nsrRand *= scale
        
        # Get the irreducible system NSR
        if self.params.highSNR:
            nsrSys = self.params.sigmaSys
        else:
            nsrSys = 10 ** (self.params.sigmaSys / 2.5) - 1
        
        # Calculate the total NSR
        nsr = np.sqrt(nsrRand**2 + nsrSys**2)
        
        return nsr

    def _calculate_m5_per_object(
        self, 
        bands: list, 
        n_objects: int, 
        per_obj_psf: dict, 
        per_obj_airmass: dict, 
        per_obj_msky: dict, 
        per_obj_tvis: dict
    ) -> np.ndarray:
        """Calculate 5-sigma limiting magnitudes per object."""
        m5_per_obj = np.zeros((n_objects, len(bands)))
        
        for i, band in enumerate(bands):
            # Start with base m5 value
            base_m5 = self._all_m5[band]
            
            # If no per-object parameters, use base value
            if (band not in per_obj_psf and band not in per_obj_airmass and 
                band not in per_obj_msky and band not in per_obj_tvis):
                m5_per_obj[:, i] = base_m5
                continue
            
            # Calculate adjustments for per-object parameters
            m5_adj = np.zeros(n_objects)
            
            # PSF adjustment
            if band in per_obj_psf:
                base_theta = self.params.theta[band]
                theta_ratio = per_obj_psf[band] / base_theta
                m5_adj += 2.5 * np.log10(0.7 / per_obj_psf[band]) - 2.5 * np.log10(0.7 / base_theta)
            
            # Airmass adjustment
            if band in per_obj_airmass:
                base_airmass = self.params.airmass[band]
                airmass_diff = per_obj_airmass[band] - base_airmass
                m5_adj -= self.params.km[band] * airmass_diff
            
            # Sky brightness adjustment
            if band in per_obj_msky:
                base_msky = self.params.msky[band]
                msky_diff = per_obj_msky[band] - base_msky
                m5_adj += 0.5 * msky_diff
            
            # Exposure time adjustment
            if band in per_obj_tvis:
                base_tvis = self.params.tvis[band]
                tvis_ratio = per_obj_tvis[band] / base_tvis
                m5_adj += 1.25 * np.log10(tvis_ratio)
            
            m5_per_obj[:, i] = base_m5 + m5_adj
        
        return m5_per_obj

    def __call__(
        self,
        catalog: pd.DataFrame,
        random_state: np.random.Generator | int | None = None,
    ) -> pd.DataFrame:
        """Calculate photometric errors for the catalog with per-object parameter support.
        
        Parameters
        ----------
        catalog : pd.DataFrame
            The input catalog of galaxies. Can include per-object parameters
            with column names like {band}_psf, {band}_airmass, etc.
        random_state : np.random.Generator, int, or None
            The random state for reproducible results.
            
        Returns
        -------
        pd.DataFrame
            Catalog with observed magnitudes and errors
        """
        # Set the rng
        rng = np.random.default_rng(random_state)

        # Get the bands we will calculate errors for
        bands = [band for band in catalog.columns if band in self._bands]

        # Get the numpy array of magnitudes
        mags = catalog[bands].to_numpy()

        # Get the semi-major and semi-minor axes
        if self.params.extendedType == "auto" or self.params.extendedType == "gaap":
            majors = catalog[self.params.majorCol].to_numpy()
            minors = catalog[self.params.minorCol].to_numpy()
        else:
            majors = None
            minors = None

        # Get observed magnitudes and errors with per-object parameter support
        obsMags, obsMagErrs = self._get_obs_and_errs_with_per_object_params(
            mags, majors, minors, bands, rng, catalog
        )

        # Handle non-detections
        if self.params.ndMode == "flag":
            # Calculate SNR
            if self.params.highSNR:
                snr = 1 / obsMagErrs
            else:
                snr = 1 / (10 ** (obsMagErrs / 2.5) - 1)

            # Flag non-finite mags and where SNR is below sigLim
            idx = (~np.isfinite(obsMags)) | (snr < self.params.sigLim)
            obsMags[idx] = self.params.ndFlag
            obsMagErrs[idx] = self.params.ndFlag

        # Save the observations in a DataFrame
        errDf = pd.DataFrame(
            obsMagErrs, columns=[f"{band}_err" for band in bands], index=catalog.index
        )
        if self.params.errLoc == "alone":
            obsCatalog = errDf
        else:
            magDf = catalog.copy()
            magDf[bands] = obsMags
            obsCatalog = pd.concat([magDf, errDf], axis=1)

        if self.params.errLoc == "after":
            # Reorder the columns so that the error columns come right after the
            # respective magnitude columns
            columns = catalog.columns.tolist()
            for band in bands:
                columns.insert(columns.index(band) + 1, f"{band}_err")
            obsCatalog = obsCatalog[columns]

        return obsCatalog

    def _get_obs_and_errs_with_per_object_params(
        self,
        mags: np.ndarray,
        majors: np.ndarray,
        minors: np.ndarray,
        bands: list,
        rng: np.random.Generator,
        catalog: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate observed magnitudes and errors with per-object parameter support."""
        # Get the NSR for all galaxies with per-object parameters
        nsr = self._get_nsr_from_mags(mags, majors, minors, bands, catalog)

        if self.params.highSNR:
            # In the high SNR approximation, mag err ~ nsr
            obsMags = rng.normal(loc=mags, scale=nsr)
        else:
            # Model errors as Gaussian in flux space
            fluxes = 10 ** (mags / -2.5)
            obsFluxes = fluxes * (1 + rng.normal(scale=nsr))
            if self.params.absFlux:
                obsFluxes = np.abs(obsFluxes)
            with np.errstate(divide="ignore"):
                obsMags = -2.5 * np.log10(np.clip(obsFluxes, 0, None))

        # If decorrelate, calculate new errors using observed mags
        if self.params.decorrelate:
            nsr = self._get_nsr_from_mags(obsMags, majors, minors, bands, catalog)

        # Handle sigLim mode
        if self.params.ndMode == "sigLim":
            with np.errstate(divide="ignore"):
                nsrLim = np.divide(1, self.params.sigLim)
            magLim = self._get_mags_from_nsr(nsrLim, majors, minors, bands)
            nsr = np.clip(nsr, 0, nsrLim)
            obsMags = np.clip(obsMags, None, magLim)

        if self.params.highSNR:
            obsMagErrs = nsr
        else:
            obsMagErrs = 2.5 * np.log10(1 + nsr)

        return obsMags, obsMagErrs