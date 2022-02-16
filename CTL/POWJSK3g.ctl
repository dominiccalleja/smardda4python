&inputfiles
 vtk_input_file='../S/SHAD_geoqm.vtk', ! shadowing geometry
 vtkres_input_file='../G/RES_geoqm.vtk', ! results geometry
 geoq_input_file='../S/SHAD_geoq.out',
 hds_input_file='../H/SHAD_hds.hds',
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
      refine_level=1,
      shadow_control=1,
      calculation_type='global',
      termination_planes = .true.,
      more_profiles = .true.
/
&termplaneparameters
      termplane_intersection=2
      termplane_direction=0
      termplane_position=0
/
&edgprofparameters
      decay_length=DECL,
      power_loss=POWe+06,
      diffusion_length=DIFL,
      profile_formula='PROF'
/
&odesparameters
      initial_dt=0.0001,
      abs_error=1.e-5,
      rel_error=1.e-5,
      termination_parameters(1)=3.,
      termination_parameters(2)=0.1,
      max_numsteps=40000,
      max_zeta=80.,
/
