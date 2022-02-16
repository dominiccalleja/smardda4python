&analysisfiles
 vtk_data_file='<TARGET>',
 vtk_input_file='<powcal_output>'
/
&miscparameters
/
&plotselections
      plot_smanalout = .true.,
      plot_anx = .true.,
      plot_ansmallx = .true.,
      plot_angnu = .true.,
/
&smanalparameters
     analysis_mode='regular'
     sort_key='Body'
     analyse_scalar='Q'
     required_statistics='intg ', 'intgsq ', 'max ', 'min ', 'maxabs', 'minabs'
     new_key='angle'
     rekey=.true.
     number_of_clusters=36
/
