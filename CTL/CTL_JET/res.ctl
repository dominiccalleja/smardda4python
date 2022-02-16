&inputfiles
 vtk_input_file='<Target>',
 eqdsk_input_file='<Equid>',
/
&miscparameters
/
&plotselections
      plot_geoqx = .true.,
      plot_geoqvolx = .false.,
      plot_geoqm = .true.,
      plot_geofldx = .true.,
      plot_gnu = .true.,
      plot_gnusil = .true.,
/
&beqparameters
      beq_fldspec=3,
      beq_cenopt=<cen_opt>,
      beq_bdryopt=<bdry_opt_r>,
      beq_psimin=<psi_min>,
      beq_psimax=<psi_max>,
      beq_psiref=<psiref>,
      beq_xiopt=2,
      beq_nzetap=<Nzetp>,
/
