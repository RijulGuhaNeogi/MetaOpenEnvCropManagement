# Scientific References

Bibliography for the WOFOST-inspired crop growth simulator used in the
Precision Agriculture Crop Management environment.

---

## Core WOFOST Model

1. **van Diepen, C.A., Wolf, J., van Keulen, H., & Rappoldt, C.** (1989).
   WOFOST: A simulation model of crop production.
   *Soil Use and Management*, 5(1), 16–24.
   https://doi.org/10.1111/j.1475-2743.1989.tb00755.x
   — *Original description of the WOFOST framework; phenology equations,
   temperature-sum approach for DVS advancement.*

2. **Boogaard, H.L., de Wit, A.J.W., te Roller, J.A., & van Diepen, C.A.**
   (2014). *WOFOST Technical Documentation (version 7.1.7)*.
   Wageningen-UR.
   — *Parameter tables (TSUM1, TSUM2, KDIF, FOTB, SLA0, LAIEM) for
   NW-European winter wheat; soil hydraulic constants.*

3. **de Wit, A., Boogaard, H., Fumagalli, D., Janssen, S., Knijn, R.,
   van Kraalingen, D., Supit, I., van der Wijngaart, R., & van Diepen, K.**
   (2019). 25 years of the WOFOST cropping systems model.
   *Agricultural Systems*, 168, 154–167.
   https://doi.org/10.1016/j.agsy.2018.06.018
   — *Updated calibrations; TSUM and SLA tables for European wheat.*

## Evapotranspiration

4. **Hargreaves, G.H. & Samani, Z.A.** (1985).
   Reference crop evapotranspiration from temperature.
   *Applied Engineering in Agriculture*, 1(2), 96–99.
   https://doi.org/10.13031/2013.26773
   — *ET₀ = 0.0023 · (T_avg + 17.8) · √ΔT · Ra · 0.408*

5. **Allen, R.G., Pereira, L.S., Raes, D., & Smith, M.** (1998).
   Crop evapotranspiration: Guidelines for computing crop water
   requirements. *FAO Irrigation and Drainage Paper 56*. Rome: FAO.
   — *Radiation conversion factor (0.408 MJ m⁻² d⁻¹ → mm d⁻¹);
   crop coefficient (Kc) methodology.*

## Light Interception & Biomass

6. **Monsi, M. & Saeki, T.** (1953).
   Über den Lichtfaktor in den Pflanzengesellschaften und seine
   Bedeutung für die Stoffproduktion.
   *Japanese Journal of Botany*, 14, 22–52.
   — *Beer–Lambert light interception: fPAR = 1 − exp(−K · LAI).*

7. **Gallagher, J.N. & Biscoe, P.V.** (1978).
   Radiation absorption, growth and yield of cereals.
   *Journal of Agricultural Science*, 91(1), 47–60.
   https://doi.org/10.1017/S0021859600056616
   — *Light-use efficiency (LUE) for C3 cereals: 2.7–3.0 g DM MJ⁻¹ PAR.*

## Water Stress

8. **Feddes, R.A., Kowalik, P.J., & Zaradny, H.** (1978).
   *Simulation of Field Water Use and Crop Yield*.
   Simulation Monographs, Pudoc, Wageningen.
   — *Piecewise-linear reduction function for transpiration under water
   deficit; basis for the θ-ratio stress model.*

## Heat Stress

9. **Prasad, P.V.V., Boote, K.J., Allen, L.H., Sheehy, J.E., &
   Thomas, J.M.G.** (2006).
   Species, ecotype and cultivar differences in spikelet fertility and
   harvest index of rice in response to high temperature stress.
   *Field Crops Research*, 95(2–3), 398–411.
   https://doi.org/10.1016/j.fcr.2005.04.008
   — *Pollen sterility above ~35 °C during flowering.*

10. **Porter, J.R. & Gawith, M.** (1999).
    Temperatures and the growth and development of wheat: A review.
    *European Journal of Agronomy*, 10(1), 23–36.
    https://doi.org/10.1016/S1161-0301(98)00047-1
    — *Cardinal temperatures for wheat phenology and pollen viability.*

11. **Wardlaw, I.F. & Moncur, L.** (1995).
    The response of wheat to high temperature following anthesis.
    I. The rate and duration of kernel filling.
    *Australian Journal of Plant Physiology*, 22(3), 391–397.
    https://doi.org/10.1071/PP9950391
    — *Kernel weight reduction under grain-fill heat stress (>32 °C).*

## Nitrogen Response

12. **Shibu, M.E., Leffelaar, P.A., van Keulen, H., & Aggarwal, P.K.**
    (2010). LINTUL3, a simulation model for nitrogen-limited situations.
    *European Journal of Agronomy*, 32(4), 255–271.
    https://doi.org/10.1016/j.eja.2010.01.003
    — *Simplified linear N-response model; N recovery fraction.*

## Soil Hydraulics

13. **Wösten, J.H.M., Pachepsky, Ya.A., & Rawls, W.J.** (2001).
    Pedotransfer functions: Bridging the gap between available basic
    soil data and missing soil hydraulic characteristics.
    *Journal of Hydrology*, 251(3–4), 123–150.
    https://doi.org/10.1016/S0022-1694(01)00464-4
    — *SMFCF, SMW pedotransfer values for clay loam, sandy loam,
    silt loam textures.*

## Regional Calibrations

14. **Aggarwal, P.K. & Kalra, N.** (1994).
    Analyzing the limitations set by climatic factors, genotype, and
    water and nitrogen availability on productivity of wheat.
    II. Climatically potential yields and management strategies.
    *Field Crops Research*, 38(2), 93–103.
    https://doi.org/10.1016/0378-4290(94)90003-5
    — *TSUM calibration for Punjab spring wheat.*

15. **Wolf, J., Hessel, R.,";"; et al.** (2011).
    *GAEZ v3.0 Global Agro-ecological Zones — Crop Parameter Database*.
    FAO/IIASA.
    — *FAO crop parameter guidelines for global calibration.*

16. **Loomis, R.S. & Connor, D.J.** (1992).
    *Crop Ecology: Productivity and Management in Agricultural Systems*.
    Cambridge University Press.
    — *Continental US wheat phenology; basis for Iowa TSUM estimates.*

## General

17. **Szeicz, G.** (1974).
    Solar radiation for plant growth.
    *Journal of Applied Ecology*, 11(2), 617–636.
    — *PAR ≈ 50 % of total incoming solar radiation.*
