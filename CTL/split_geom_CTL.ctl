&miscparameters
 option='panel',
 max_number_of_panels=1200,
 max_number_of_transforms=40,
 new_controls=.TRUE.
/
&vtktfmparameters
 option='panel',
 max_number_of_panels=1200,
 max_number_of_transforms=40,
 angle_units='degree',
 make_same=.FALSE.,
 split_file=.TRUE.,
/
&vtkfiles
 vtk_input_file=<FILE_NAME>, 
/
&panelarrayparameters
      panel_bodies=1,
      panel_transform=1,
/
&positionparameters
      position_transform = 42,
      position_offset=       0.00000000, 00, 0,
/

