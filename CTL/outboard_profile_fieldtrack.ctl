&inputfiles
geoq_input_file='<run_id>RES_geoq.out',
hds_input_file='<run_id>HDS_hds.hds',
vtk_input_file='<run_id>SHAD_geoqm.vtk', ! shadowing geometry
vtkres_input_file='<run_id>RES_geoqm.vtk', ! results geometry
/
&miscparameters
/
&plotselections
      plot_powx = .true.,
      plot_flinx = .true.,
      plot_flinm = .true.,
      plot_flinends = .true.,
/
&powcalparameters
      refine_level=2,
      shadow_control=1,
      calculation_type='global',
      skylight=.false.
      termination_planes = .true.,
      more_profiles = .true.
      number_of_tracks = 42,
      element_numbers = 50812,50963,51079,51195,51196,51362,51454,51459,51461,51526,51533,51546,51676,51682,51688,51689,51799,51801,51802,51804,51812,51925,51927,51976,51994,51995,52154,52168,52169,52286,52298,52299,52300,52301,52476,52738,52895,52897,52898,52900,53160,53161,
/
&termplaneparameters
      termplane_intersection=2
      termplane_direction=0
      termplane_position=0
/
&edgprofparameters
      decay_length=0.00103,
      power_loss=1E7,
      diffusion_length=0.0004,
      profile_formula='eich'
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
