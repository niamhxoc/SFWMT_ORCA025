import warnings,glob,os,time
import sys
sys.path.append("/home/users/nboc20/jupyter_notebooks/packages")
import xarray as xr         
import numpy as np
import time as timer
import regionmask
import gsw
import pop_tools
warnings.filterwarnings('ignore')

# define any constants needed...
cpsw=3.996e3          ## specific heat capacity of sea water: J/kg/degC
datapath = '/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/Omon/'

def main():
    
    # data import
    area = xr.open_dataset('/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/Ofx/areacello/gn/' \
                           'latest/areacello_Ofx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc')['areacello'][850:,:]
    basin = xr.open_dataset('/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/Ofx/basin/gn/' \
                            'latest/basin_Ofx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc')['basin'][850:,:]
    basin_mask = xr.open_dataset('/gws/pw/j07/wishbone/nboc20/outputs/north_atl_mask.nc').mask[850:,:]

    def get_wmt(method, init_year, space, ens_mem):
        """
        calculates the surface forced water mass transformation of HadGEM3-GC3.1-MM on JASMIN.

        If the argument 'outcrops' is passed then the surface density contribution will be calculated with fixed fluxes
        If the argument 'fluxes' is passed then the surface flux contribution will be calculated with fixed surface density
        

        Parameters
        ----------
        method : str
            specifies which version of sfwmt to calculate

        year, init_year, ens_mem : int
            specifies the file to calculate

        space : int
            specifies the latitude or area to calculate the sfwmt over

        Returns
        -------
        sdenq : arr
            the surface density flux due to heat
            
        sdenf : arr
            the surface density flux due to freshwater

        ssd : arr
            the surface density array

        """
        
        # data names
        variables = ['tos','sos','hfds']
        
        ds = xr.Dataset()
        if (space >= 1) & (space <= 5):
            # import data and use area mask
            # Values: 1 Arctic, 2 Labrador, 3 Nordic, 4 Irminger, 5 East SPG
            for var in variables:
                ds[var] = xr.open_mfdataset(f'/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s{init_year}-r{ens_mem}i1p1f2/Omon/'+var+'/gn/latest/'+var+f'_Omon_HadGEM3-GC31-MM_dcppA-hindcast_s{init_year}-r{ens_mem}i1p1f2_gn_*.nc')[var][:,850:1205,:].where(basin_mask==space).where(basin.latitude>=50)

            ds['wfo'] = xr.open_mfdataset(f'/gws/ssde/j25b/canari/users/nboc20/outputs/hindcasts/wfo/s{init_year}/r0{ens_mem:02}i1p1f2/wfo_Omon_HadGEM3-GC31-MM_dcppA-hindcast_s{init_year}-r{ens_mem}i1p1f2_gn_*.nc')['wfo'][:,850:1205,:].where(basin_mask==space).where(basin.latitude>=50) #import wfo from canari gws


        else:
            # import data and use latitude mask
            for var in variables:
                ds[var] = xr.open_mfdataset(f'/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s{init_year}-r{ens_mem}i1p1f2/Omon/'+var+'/gn/latest/'+var+f'_Omon_HadGEM3-GC31-MM_dcppA-hindcast_s{init_year}-r{ens_mem}i1p1f2_gn_*.nc')[var][:,850:1205,:].where(((basin==2)|(basin==4))).where(basin.latitude>=space)
                ds['wfo'] = xr.open_mfdataset(f'/gws/ssde/j25b/canari/users/nboc20/outputs/hindcasts/wfo/s{init_year}/r0{ens_mem:02}i1p1f2/wfo_Omon_HadGEM3-GC31-MM_dcppA-hindcast_s{init_year}-r{ens_mem}i1p1f2_gn_*.nc')['wfo'][:,850:1205,:].where(((basin==2)|(basin==4))).where(basin.latitude>=space) #import wfo from canari gws
            name = space

        # assign variables
        time=ds['time']
        tos=ds['tos']
        sos=ds['sos']
        depth=0
        p=0

        # set climatologies for either density or flux driven sfwmt
        n_years=np.ceil(np.shape(ds.sos)[0]/12.0).astype(int)
        if method == 'outcrops':
            ds.hfds.values=np.tile(ds.hfds.groupby("time.month").mean('time').data,(n_years+1,1,1))[10:-9,:,:]
            ds.wfo.values=np.tile(ds.wfo.groupby("time.month").mean('time').data,(n_years+1,1,1))[10:-9,:,:]

        elif method == 'fluxes':
            ds.sos.values=np.tile(ds.sos.groupby("time.month").mean('time').data,(n_years+1,1,1))[10:-9,:,:]
            ds.tos.values=np.tile(ds.tos.groupby("time.month").mean('time').data,(n_years+1,1,1))[10:-9,:,:]
            
        ### teos10
        #ct = gsw.CT_from_t(sos, tos, p) # conservative temperature
        #sa = gsw.conversions.SA_from_SP(sos, p, ds['longitude'], ds['latitude']) # absolute salinity
        #ssd = gsw.density.sigma2(sa, ct) # sigma_2
        #ssd.values[np.isnan(ssd.values)]=0.
        #alpha = gsw.alpha(sa, ct, p) # thermal expansion coefficient 1/C
        #beta = gsw.beta(sa, ct, p) # haline contraction coefficient g/kg
        
        ### eos80
        rho,drhods,drhodt = pop_tools.eos(salt=sos,temp=tos,return_coefs=True,depth=depth)
        alpha = (drhodt/rho)*-1		# 1/degC
        beta =  (drhods/rho)		# kg(Seawater)/kg(salt)
        ssd = pop_tools.eos(salt=sos,temp=tos,return_coefs=False,depth=depth+2000.)-1000
        ssd.values[np.isnan(ssd.values)]=0.

        # calculate tendencies
        sdenq=(-alpha*ds['hfds']/cpsw).astype(np.float32); sdenq.values[np.isnan(sdenq.values)]=0.
        sdenf=(-beta*ds['wfo']*(0.001*sos/(1-0.001*sos))).astype(np.float32); sdenf.values[np.isnan(sdenf.values)]=0.
        print('calculated heat and freshwater tendencies')

        return sdenq, sdenf, ssd
    
    ### sfwmt calculation settings
    time1=time.time()
    file_name = sys.argv[1]
    #file_name=file_name[-16:-12]
    lat = int(sys.argv[2])
    print(f'space is {lat}')
    # Values: 1 Arctic, 2 Labrador, 3 Nordic, 4 Irminger, 5 East SPG
    space_str = ['arctic', 'labrador', 'nordic', 'irminger', 'east']
    version = int(sys.argv[3])
    if version==1:
        method = 'main'
    elif version==2:
        method = 'outcrops'
    elif version==3:
        method = 'fluxes'
    print(f'version is {version} {method}')
    init_year = int(sys.argv[4])
    print(f'initialisation year is {init_year}')
    ens_mem = int(sys.argv[5])
    print(f'ensemble member is {ens_mem}')
    
    ### run code
    sdenq, sdenf, ssd = get_wmt(method, init_year, lat, ens_mem)

    ## create the density space we'd like to calculate WMT on. 30-38 kg m-3, with a bin size of 0.05
    sig = np.linspace(30.025,37.875,158); sig_lo = sig-0.025; sig_hi = sig+0.025; dsig=sig_hi-sig_lo

    
    sigma2=xr.DataArray(sig,dims=['sigma'],coords={'sigma':sig},
        attrs={'standard_name':'sigma2','units':'kg m-3'})
    sigma2.encoding['_FillValue']=None;
    print('calculated sigma')
     
    # set up output xarrays;
    nt = len(ssd.time); ns=len(sig); ny = len(ssd.j); nx = len(ssd.i)
    tlon=ssd['i']; tlat=ssd['j'];
    dims=['time','sigma']
    
    WMT_Q = xr.DataArray(data=np.zeros([nt,ns]),
        dims=dims,
        coords={'time':ssd.time,'sigma':sigma2},
        name='wmt_q',attrs={'long_name':'Water Mass Transformation (heat)','units':'Sv'})
    WMT_Q.encoding['_FillValue']=1.e30
    
    WMT_F = xr.DataArray(data=np.zeros([nt,ns]),
        dims=dims,
        coords={'time':ssd.time,'sigma':sigma2},
        name='wmt_f',attrs={'long_name':'Water Mass Transformation (freshwater)','units':'Sv'})  
    WMT_F.encoding['_FillValue']=1.e30

    SFWMT = xr.DataArray(data=np.zeros([nt,ns]),
        dims=dims,
        coords={'time':ssd.time,'sigma':sigma2},
        name='sfwmt',attrs={'long_name':'Water Mass Transformation','units':'Sv'})  
    WMT_F.encoding['_FillValue']=1.e30
    
    ## loop over each density bin (from high to low densities) and calculate the outcropping region, then aggregate wmt fluxes
    for isig in range(ns):
        outcrop_domain = ((ssd.values>sig_lo[isig]) & (ssd.values<=sig_hi[isig]))*1 
        print(sig[isig])
        WMT_Q.values[:,isig]= np.nansum(((np.multiply(sdenq,outcrop_domain)/dsig[isig])*area),axis=(-1,-2))*1e-6  # Sv
        WMT_F.values[:,isig]= np.nansum(((np.multiply(sdenf,outcrop_domain)/dsig[isig])*area),axis=(-1,-2))*1e-6  # Sv

    SFWMT.values = WMT_Q + WMT_F
        
    if lat < 6:
        space_name = space_str[lat-1]
        #WMT_Q.to_netcdf(f'/gws/nopw/j04/canari/users/nboc20/outputs/hindcasts/sfwmt_smaller_bin/s{init_year}/qwmt_area_{space_name}_{file_name}_r{ens_mem}.nc',format="NETCDF4", engine="netcdf4")
        #WMT_F.to_netcdf(f'/gws/nopw/j04/canari/users/nboc20/outputs/hindcasts/sfwmt_smaller_bin/s{init_year}/fwmt_area_{space_name}_{file_name}_r{ens_mem}.nc',format="NETCDF4", engine="netcdf4")
        SFWMT.to_netcdf(f'/work/scratch-pw5/nboc20/hindc_sfwmt/s{init_year}/sfwmt_{method}_area_{space_name}_s{init_year}_r{ens_mem}.nc',format="NETCDF4", engine="netcdf4")
    
    else:    
        #WMT_Q.to_netcdf(f'/gws/nopw/j04/canari/users/nboc20/outputs/hindcasts/sfwmt_smaller_bin/s{init_year}/qwmt_area_{lat}_{file_name}_r{ens_mem}.nc',format="NETCDF4", engine="netcdf4")
        #WMT_F.to_netcdf(f'/gws/nopw/j04/canari/users/nboc20/outputs/hindcasts/sfwmt_smaller_bin/s{init_year}/fwmt_area_{lat}_{file_name}_r{ens_mem}.nc',format="NETCDF4", engine="netcdf4")
        
        SFWMT.to_netcdf(f'/work/scratch-pw5/nboc20/hindc_sfwmt/s{init_year}/sfwmt_{method}_{lat}_s{init_year}_r{ens_mem}.nc',format="NETCDF4", engine="netcdf4")
        
    print(f'DONE creating /work/scratch-pw5/nboc20/hindc_sfwmt/s{init_year}/sfwmt_{method}_{lat}_s{init_year}_r{ens_mem}.nc','. Total time = ',timer.time()-time1,'s')
    
    return True

if __name__=="__main__":
    main()
