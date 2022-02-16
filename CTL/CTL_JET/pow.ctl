&inputfiles
geoq_input_file='<SimName>RES_geoq.out',
hds_input_file='<SimName>HDS_hds.hds',
vtk_input_file='<SimName>SHAD_geoqm.vtk', ! shadowing geometry
vtkres_input_file='<SimName>RES_geoqm.vtk', ! results geometry
/
&miscparameters
/
&plotselections
      plot_powx = .true.,
      plot_flinx = .false.,
      plot_flinm = .false.,
      plot_flinends = .false.,
/
&powcalparameters
      refine_level=2,
      shadow_control=1,
      calculation_type='global',
      skylight=.false.
      termination_planes = .true.,
      more_profiles = .true.
/
&termplaneparameters
      termplane_intersection=2
      termplane_direction=0
      termplane_position=0
/
&edgprofparameters
      decay_length=<DECL>,
      power_loss=<POW>,
      diffusion_length=<DIFL>,
      profile_formula='<PROF>'
/
&odesparameters
      initial_dt=1.e-5,
      abs_error=1.e-5,
      rel_error=1.e-5,
      termination_parameters(1)=3.,
      termination_parameters(2)=0.1,
      max_numsteps=40000,
      max_zeta=80.,
/
