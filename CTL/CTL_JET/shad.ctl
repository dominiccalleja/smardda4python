&inputfiles
 vtk_input_file='<Shadow>',
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
      plot_eqdsk_boundary = .true.
/
&beqparameters
      beq_fldspec=3,
      beq_cenopt=<cen_opt>,
      beq_bdryopt=<bdry_opt_s>,
      beq_psimin=<psi_min>,
      beq_psimax=<psi_max>,
      beq_xiopt=2,
      beq_nzetap=<Nzetp>,
/
