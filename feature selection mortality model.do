cd "C:\Users\bogwel\OneDrive\Billy\School\PhD\Analysis\Data"

u "C:\Users\bogwel\OneDrive\Billy\School\PhD\Analysis\Mortality\Data\ipd_all_results_09Jun2021.dta", clear
keep pagelink ipdadseps ipdchivres
replace ipdadseps=trim(ipdadseps)
replace ipdchivres=. if !inlist(ipdchivres,1,2)
duplicates list pagelink
duplicates drop pagelink, force
sa ipd_hiv_sep, replace



u rota_SCRH_08Dec2020, clear
merge 1:1 pagelink using ipd_hiv_sep
keep if _merge==3
drop _merge

ta ipdchivres
ta ipdchivres death, row chi2 exact

summ age_mo, det

gen year=year(ipddate)
ta death

gen diffdate=ipdddis-ipddate
summ diffdate if death==1, det

ta agecat death, col chi2 exact
ta ipdsex death, col chi2

foreach var of varlist age_mo ipdtem ipdhg ipddresp ipdpuls{
				bysort death: summ `var', det 
					ranksum `var',by(death)
								}

//sympduration1								

foreach var of varlist ipdmvom ipdvomev ipdmcon ipdletha ipdmuncons ipdmfev ipdpneg ipdmwgt ipdundk ipdalert ipdresirit ipdconvul ipdsunkeye ipdskinp ipdcaprefi ipddrink ipdbulgf ipdsunkf ipdreyes ipdwast  ipdtfiv ipdtorsp dehydrationstatus vesikari_cat  stunting nutri_wasting underweight ipdmadmin ipdhu ipdchestin ipdstrid ipdnasal {
						ta `var' death, col chi2 exact
									}

//ipdmbld 

foreach var of varlist ipdmdrhd drhmax maxvom vesikari_int ipdlbpar bgluco{
				bysort death: summ `var', det 
					ranksum `var',by(death)
								}



//ipdtem ipdhg ipdpuls ipdmvom ipdletha ipdmuncons ipdmfev ipdpneg ipdmwgt ipdalert ipdresirit ipdsunkeye ipdskinp ipdcaprefi ipddrink ipdbulgf ipdsunkf ipdreyes ipdwast ipdtfiv dehydrationstatus stunting underweight nutri_wasting ipdmadmin ipdhu ipdchestin ipdstrid ipdnasal ipdmdrhd maxvom vesikari_int

corr stunting nutri_wasting underweight ipdwast ipdmadmin ipdhu
corr ipdmfev ipdtem
corr ipdletha ipdmuncons ipdalert ipdresirit ipdsunkeye ipdskinp ipddrink dehydrationstatus
//drop underweight correlated with wasting and stunting
//drop ipdwast as well
//ipdadmdehy correlated with dehydrationstatus


//agecat,ipdtem,ipdhg,ipdpuls,ipdmvom,ipdletha,ipdmuncons,ipdmfev,ipdpneg,ipdmwgt,
                                 ipdalert,ipdresirit,ipdsunkeye,ipdskinp,ipdcaprefi,ipddrink,ipdbulgf,ipdsunkf,
                                 ipdreyes,ipdtfiv,dehydrationstatus,stunting,nutri_wasting,ipdmadmin,ipdhu,
                                 ipdchestin,ipdstrid,ipdnasal,ipdmdrhd,maxvom,vesikari_int,death,
                                 ipdvomev,ipdmcon,ipdconvul,ipdundk,vesikari_cat

foreach var of varlist ipdsunkeye ipdskinp ipdcaprefi dehydrationstatus {
    foreach v of varlist ipdsunkeye ipdskinp ipdcaprefi dehydrationstatus {
	    ta `var' `v', chi2 V col
	}
}

//Trend test
ta year death  
//nptrend death,by (year)
import excel using "C:\Users\bogwel\OneDrive\Billy\School\PhD\Analysis\Mortality\Data\Trend_data.xlsx", firstrow clear
ptrend Death Survived Year
