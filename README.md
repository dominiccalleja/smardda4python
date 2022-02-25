# smardda4python

The core of this library is a python wrapper around the fortran SMARDDA-PFC software, with a whole array of
uncertainty quantification workflows included to conduct footprint model calibration, validation,
PFC displacement studies, time-series analyses, develop surrogate models for plasma power
deposition profiles, and to disentangle the effect of transient and steady state power deposition,
and to compute temperatures from the SMARDDA-PFC simulations.

\texttt{smardda4python} is the core (main class) of the software library developed within this work, and it is the wider library name. \texttt{smardda4python} is a class which builds a constructor for a single \texttt{SMARDDA-PFC} evaluation. The use of this class is very similar to that one would expect from standard interaction with the \texttt{SMARDDA-PFC} core engine. This class simplifies the task of constructing a \texttt{SMARDDA-PFC} simulation, and many users will find a python interface much more familiar than having to modify specific input files. Some effort has been invested in organising evaluation parameters into a more natural structure than the original input files require. The parallel implementation for \texttt{smardda4python} inherits all attributes of the standard class, a standard class can be passed to it also, with some additional functionality to evaluate large numbers of simulations. It is useful for Monte-Carlo or other numerical uncertainty analyses that typically require a large number of model evaluations.  


\subsubsection{Modules \texttt{smardda4python}}

\paragraph{\texttt{PPF\_Data}}

For JET, much of the complexities of the experimental data management were solved through the introduction of the Simple Access Layer (SAL) in 2018 \cite{SAL}. The \texttt{PPF\_Data} module in \texttt{smardda4python} manages the necessary data sources via the SAL tools. The \texttt{PPF\_Data} class requires a user to specify the JET pulse number it then calls the SAL library and pulls all the necessary data sources that may be needed by any of the other classes. 

\paragraph{\texttt{TStools}}

The \texttt{TStools} module contains a variety of tools for developing time series analyses within the \texttt{smardda4python} framework. The library contains a variety of functions to handle the heavy lifting of developing time-series simulations, including manipulating directory structures and file names.

\paragraph{\texttt{SMARDDA\_FWmod}}

\texttt{FWmod} is a collection of functions for manipulating vtk files. It can handle any translation, scale, and some deformations, it does so by copying, modifying and rewriting a specified .vtk file. It contains all the tools necessary to manipulate geometries for the purposes of first wall misalignment studies.

\begin{figure}
    \centering
    \includegraphics[width=0.7\textwidth]{finaltranslationimage.png}    
    \caption{Examples of body translations of PFCs with \texttt{SMARDDA\_FWmod}}
    \label{fig:tile_translation}
\end{figure}


\paragraph{\texttt{SMARDDA\_Analysis}} 

\texttt{SMARDDA\_Analysis} has a single class, \texttt{Analysis}. The analysis class contains a selection of methods to extract and format the results of a \texttt{SMARDDA-PFC} powcal output. There are several functions to support different workflows, include mapping the vtk outputs to \texttt{ANSYS mesh}, and formatting the radial power deposition profile for the \texttt{SMARDDA\_surrogate} and \texttt{SMARDDA\_heatequation} tools. 

\texttt{SMARDDA-PFC} utilises the VTK format. There are several advantages to the use of VTK formats, they are easily interpreted plain text, they utilise a simple compact format with consistency in representation across all modules within \texttt{SMARDDA-PFC}, and they are easily managed and modified. \texttt{SMARDDA\_Analysis} has been developed to manage the \texttt{smardda4python} simulation result outputs. The module contains a variety of tools for managing larger VTK results files. The core \texttt{Analysis} class manages the surface power deposition (powx.vtk), and boundary field files (geofldx.vtk) for \texttt{SMARDDA-PFC} simulations. 

The \texttt{Analysis} class extracts the results geometry, and power statistics. And, includes several functions to compute power statistics for \\ \texttt{SMARDDA-PFC} results such as hotspot locations, integrating across individual surfaces or bodies within composite geometries, and for calculating parallel power deposition statistics. 

\paragraph{\texttt{SMARDDA\_surrogate}}

Typically, evaluations of \texttt{SMARDDA-PFC} workflows take several minutes. For large scale UQ analyses this is prohibitively expensive, where typically depending on the complexity of the study, model evaluations in the order of hundreds or thousands might be required. A common strategy to deal with this challenge is to deploy a surrogate model in the place of the computationally expensive physics-based model. Within the \texttt{smardda4python} framework a Gaussian Process is used to emulate the radial power deposition. This type of model was introduced in Section \ref{sec:surrogate_modelling_intro}. Here, the computational implementation will be briefly discussed. The mathematical construction of the GPE with optimal kernel selection for emulation of the radial heat deposition profile of the JET outboard divertor is discussed in Section \ref{sec:boundary condition GP}.

The GPy \cite{GPySheffeild} module in python, which remains under constant development and is a leader in the class, is implemented in the \texttt{SMARDDA\_surrogate} module. The functionality of the GPy tools are inherited in the \texttt{smardda4python} library, with a reduced set of methods tailored for utility in the \texttt{SMARDDA-PFC} analyses workflows. Some effort has been invested in developing a robust kernel selection, optimisation, and data prepossessing procedures for good emulation of footprint profiles. These configurations are offered as default for \texttt{SMARDDA-PFC} UQ workflows. \texttt{smardda4python} in tern benefits from the robust and efficient optimisation modules within GPy. 

The naive construction of the GPE training and test data within the \texttt{smardda4python} framework is handled in the core module. The post processing for each simulation is handled in the \texttt{SMARDDA\_Analysis} module, and the preparation of the training data is conducted in the \texttt{SMARDDA\_surrogate} module.




Some diagnostics for a GPE developed for JPN 91575 are shown in Figure \ref{fig:gp_profile} and \ref{fig:gp_max}. To develop this GPE a 187 sample exploration of $\lambda_q$ and $S_q$ was conducted with a Latin Hypercube Sample of the parameter space. 30 radial sample points of the \texttt{SMARDDA-PFC} radial profile are also passed as input to the GPE. The GPE is then trained on 40\% of data for the response variable, scaled Q at the radial locations. Four different realisations are evaluated against the testing data for $\lambda_q$ and $S_q$ values close to the region of greatest interest in Figure \ref{fig:gp_profile}. These results are indicative of the majority of the parameter domain as can be seen in Figure \ref{fig:gp_max} where the summary metric on the left plot is chosen to be maximum of the simulated vs predicted surface power values.   

\paragraph{\texttt{SMARDDA\_heatequation}} 

The \texttt{SMARDDA\_heatequation} module contains two heat equation solvers, one finite difference solver and a finite element solver, see Section \ref{modelling_section}. Both solve the heat equation for a bulk material CFC rectangular cross-section approximation to the tile 6 JET divertor.

The motivation for employing the reduced order explicit finite difference solver on a simplified thermal diffusion model is to have a computationally tractable model for UQ workflows. Again, similarly to the motivation for the development of the \texttt{SMARDDA\_surrogate} this necessitates being able to evaluate the model relatively cheaply. This model has been utilised in the average footprint parameter updating, see Section \ref{initial_updating_section}. This model performs well for the updating of the average, smooth, boundary condition parameters $\lambda_q$ and $S_q$ as in the temperature targeted updating workflows. The model is also written entirely in python, so it doesn't require users to install any additional software to use it. 

Additionally, the second finite element solver has also been implemented to support the analysis of workflows with irregular boundary conditions. This is necessary for solving the heat equation for the much more challenging irregular temporal heat flux profiles that are experienced by the divertor in ELMy discharges. The model has been constructed and solved with the FEniCS software \cite{AlnaesLoggEtAl2009a,LoggWells2010a,AlnaesBlechta2015a}. A full description of the model can be seen in Section \ref{heat_equation}. To resolve ELMs a 6 second pulse is typically evaluated with an explicit finite difference temporal discretisation the order of 30,000, and around 200 elements in the spatial mesh. 

Both models, and their constituent building blocks, are wrapped in the \texttt{SMARDDA\_heatequation} class. Three validation cases are discussed in Section \ref{heat_equation}. Some of the early development in the work presented in this thesis consisted of developing an ANSYS model to compute the evolution of full-3D surface temperature rise in JET plasma facing components. There is nothing that prevents FEniCS to be used for this purpose, and its efficacy in the 2D simplified case is very encouraging. It is intended the full-field solver will be implemented in the future.

\paragraph{\texttt{SMARDDA\_updating}} 

The \texttt{smardda\_updating} tools contain a number of algorithms and tools to conduct Approximate Bayesian Inference \cite{Toni2009,Refregier,Marin2009} on $\lambda_{q}$, $S_q$ and $P_{Sol}$. The library contains an optimised implementations of the Transitional Markov Chain MCMC algorithm \cite{Ching2007,BeerBrodgie2019,Betz2016}. The library also contains a variety of stochastic distance metrics.

The TMCMC algorithm, discussed in Section \ref{TMCMC description} is implemented in the \texttt{SMARDDA\_updating} module. As was discussed the updating algorithm supports scaling across large numbers of cpus because of the invocation of independent Markov Chains. The implementation in this case however is complicated by the data requirements of each worker. The type of studies discussed within this thesis have two large memory requirements, the surrogate model and the IR camera measurements used as the target data in the updating procedure. 

A naive implementation of an updating code like the TMCMC runs into the challenge that typically parallelisation libraries have several well known limitations for modern computing tasks. The concept of classes and functions that are typical in serial programming don't usual translate to multiple processors. So, whilst standard parallisation libraries like multiprocessing and pathos tend to serialise objects that are to be evaluated on remote workers, and then deserialise them at run time. This process can create a large overhead, particularly when there are large amounts of data to continually serialise and reload. The \texttt{smardda4python} implementation doesn't generate that excess overhead thanks to a novel solution implementation.

There is one feature of TMCMC that helps in this endeavour also that has not previously been discussed in the literature. As was described in Section \ref{sec:MCMC introduction}, MCMC algorithms aren't typically parallisable in an efficient manner since algorithm efficiency requires the processors to communicate, but processor communication generates computational overheads, particularly where work is not evenly distributed. The resampling process, as opposed to kernel density sampling, means its possible to seed chains with an inverse transform sample of the cumulative distribution function of the weights at the beginning of the independent MH evaluations, $F_{\hat{\phi}_{(i,j)}}$. The length of chains is then the difference between the indicies of the adjacent samples of $F_{\hat{\phi}_{(i,j)}}$. This means one of the biggest determinants of work is known prior to sending tasks to remote workers. This adds additional opportunities to distribute the work across processors evenly during execution by packaging tasks reducing excess idle time by disproportionate work allocations. 

The next novelty of the \texttt{smardda4python} is the use of the Ray multiprocessing library. Ray - originally developed for machine learning applications - accomplishes parallelisation by turning the serial concept of a class into a multiple cpu corollary called the Ray remote. The implementation, shown in Figure \ref{fig:TMCMC_algo}, shows how this has been exploited to reduce the computational burden of repeatedly loading large amounts of data. When the \texttt{SMARDDA\_updating} class is instantiated all the models and data can be loaded independently on each remote worker. The worker that then waits for the task from the head node. This initialisation, with the large data loading overhead, need only happen once on each worker for the full evaluation of the TMCMC analysis. In native python its not possible to accomplish this with other parallelisation tools. Loading the data for the full TMCMC implementation only once has the additional benefit of significantly reducing the amount of data transfers required between nodes also. Which means only the seed samples for each worker need be passed between the head and remotes during the evaluation. This module has not been tested on cloud architecture, but Ray is optimised for it, so it should scale to those sorts of architectures.   




\paragraph{\texttt{SMARDDA\_ELMs}} 

The \texttt{SMARDDA\_ELMs} module contains a variety of tools to implement the disentanglement of the transient and steady-state power deposition workflow described in Section \ref{ELMS_Section}. This workflow is not only a novel stochastic approach to disentangling the ELM and inter-ELM power exhaust, and there by inferring the steady-state power entering the SOL, but also as a by-product generates a stochastic model for the ELM induced power deposition on the outboard divertor. 

This module implements the very simple ELM shape model inferred from the IR camera field measurements, see Section \ref{stochastic elm model}. Its a trivial task to modify or extend this model to either incorporate other diagnostics or to fit a characteristic shape model to other regions of the vessel. This then provides a generative stochastic model to predict the effect of ELMs on surface temperatures, see Section \ref{Sec:Full Temp Prediction}. 

Inside \texttt{SMARDDA\_ELMs} there are two classes, one for the characterisation of the ELM stochastic model, the other for the back calculation of $P_{SOL}$. This module also includes a variety of analytical and simulation forward and back propagation uncertainty quantification algorithms. A description of the error terms for the respective JET diagnostics is described in Section \ref{sec:jet diagnostics}. A comprehensive set of analysis tools include methods to characterise the model error, in this case the model error has two components, error on the regression coefficients of the ELM shape model, and the inferential uncertainty because of the limited number of ELMs in any particular pulse. The second source of uncertainty characterised is the stochasticity related to the variability in the ELM behaviour which is characterised by the Gaussian Mixture Model described in Section \ref{ELMS_Section}. And the measurement uncertainty is characterised as an interval on each measurement and integrated into the calculation of $P_{SOL}$ trough and integrated backward calculation. 

Most of the uncertainties on measurements from the JET PPF repository are either assumed or where possible have been taken from the repository themselves. Users are able to specify different uncertainties on each measurement utilised in the analysis. This offers a whole array of tools to also conduct sensitivity studies based on these uncertainties, see Section \ref{var sensitivity} and \ref{pinch sensitivity}. This could be deployed in the future for a variety of different analyses. For example, perhaps in a future round of JET upgrades there may be a number of options about diagnostics systems that may benefit from an upgrade. If one of the objectives is to reduce uncertainties, and resources are limited, the tools developed within the library can help to direct that resource where it is likely to have the greatest impact on reducing uncertainties for some quantity of interest. 
