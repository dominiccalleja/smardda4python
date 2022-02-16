&inputfiles
 vtk_input_file='<Shadow>',
 eqdsk_input_file='<Equid>'
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
      beq_psibig=1,
      beq_cenopt=2,
      beq_bdryopt=9,
      beq_psiref=-0.0304,
      beq_xiopt=2,
      beq_nzetap=<Nzetp>,
      equil_helicity_ok=.true.
/
