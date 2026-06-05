% Real-Time Calculus (RTC) Toolbox
%
% Copyright (c) 2004-2007
% Computer Engineering and Networks Laboratory (TIK)
% ETH Zurich, Switzerland.
% All rights reserved.
% For support, write to: rtc@tik.ee.ethz.ch
%
% 
% Modular Performance Analysis Toolbox
% ====================================
%
% Construct Curves
% ----------------
%   rtcbd         - Create the curve set of a bounded delay resource model.
%   rtcbdl        - Create the lower curve of a bounded delay resource model.
%   rtcbdu        - Create the upper curve of a bounded delay resource model.
%   rtcbdbs       - Create a bounded delay curve set with bounded slope.
%   rtcbdbsu      - Creates a bounded delay upper curve with bounded slope.
%   rtcbdbsl      - Creates a bounded delay lower curve with bounded slope.
%   rtccurve      - Creates an arbitrary curve.
%   rtcfs         - Create the curve set to a full service resource.
%   rtcfsl        - Create the lower curve to a full service resource.
%   rtcfsu        - Create the upper curve to a full service resource.
%   rtcmd         - Create the curve set of a maximum delay resource model.
%   rtcpjd        - Create the curve set to a (P,J,D) event model.
%   rtcpjdl       - Create the lower curve to a (P,J,D) event model.
%   rtcpjdu       - Create the upper curve to a (P,J,D) event model.
%   rtcps         - Create the curve set to a periodic resource model.
%   rtcpsl        - Create the lower curve to a periodic resource model.
%   rtcpsu        - Create the upper curve to a periodic resource model.
%   rtctdma       - Create the curve set to a TDMA resource model.
%   rtctdmal      - Create the lower curve to a TDMA resource model.
%   rtctdmau      - Create the upper curve to a TDMA resource model.
%
% Plot Curves
% -----------
%   rtcplot       - Plot a curve or curve set.
%   rtcplotbounds - Plot the upper and lower bounds to a curve or curve set.
%   rtcplotc      - Plot a curve or curve set.
%   rtcploth      - Plot maximum horizontal distance b. two curves or curve sets.
%   rtcplotv      - Plot the max vertical distance bet. two curves or curve sets.
%   rtcmatplot    - Performs the plot of a matrix of curves.
%
% Basic Operators
% ---------------
%   rtcaffine     - An affine argument transformation of a curve.
%   rtcceil       - Ceil of a curve or curve set.
%   rtcclone      - Creates a clone of a cuve or curve set.
%   rtceq         - Equality check of two curves or curve sets.
%   rtcexport     - Exports a curve or a curve set to a textual representation.
%   rtcfloor      - Floor of a curve or curve set.
%   rtch          - Compute the max horizontal distance betw two curves or curve sets.
%   rtcmatmin     - Performs the minimum of two matrices.
%   rtcmax        - Maximum of two curves, curve sets or curves and numbers.
%   rtcmin        - Minimum of two curves, curve sets or curves and numbers.
%   rtcminus      - Substraction of two curves or curve sets.
%   rtcplus       - Addition of two curves or curve sets.
%   rtcrdivide    - Division of a curve or curve set with a scalar.
%   rtcscale      - Affine value transformation of a curve or curve set.
%   rtcsteps      - Step curves (floor and ceil) of a curve set.
%   rtctimes      - Multiplication of a curve or curve set with a scalar.
%   rtcuminus     - Unary minus of a curve or curve set.
%   rtcuplus      - Unary plus of a curve or curve set creates a clone.
%   rtcv          - Compute the max vertical distance bet. two curves or curve sets.
%   rtcvalue      - Determines the value of a curve at some point
%
% Complex Operators
% -----------------
%   rtcapprox     - Create curves approximating a given set of curves.
%   rtcapproxs    - Create a curve approximating a given curve.
%   rtcconcat     - Computes the concatenation of curves or curve sets.
%   rtcconvpjd    - Compute 'pjd' parameters for curves approximating a set of curves.
%   rtcinvert     - Inverts a curve or curve set.
%   rtcmatminconv - Performs the min-convolution of two matrices.
%   rtcmatminclos - Performs the min-closure of the input matrix.
%   rtcmaxconv    - Max-plus convolution of curves.
%   rtcmaxdeconv  - Max-plus deconvolution of curves.
%   rtcminconv    - Max-plus convolution of curves.
%   rtcmindeconv  - Max-plus deconvolution of curves.
%   rtctighten    - Tightens a curve set using additivity.
%
% MPA Components
% --------------
%   rtcand        - Compute the output of a component with AND activation.
%   rtcbuf        - Compute the buffer requirement at a GPC.
%   rtcdel        - Compute the maximum delay at a GPC.
%   rtcedf        - Compute the output of an EDF processing component.
%   rtcfifo       - Compute the output of an FIFO processing component.
%   rtcfork       - Computes the decomposition of a structured stream into simple substreams.
%   rtcforkstruct - Computes the decomposition of a structured stream into structured substreams.
%   rtcgpc        - Compute the output of a greedy processing component.
%   rtcgsc        - Compute the output of a greedy shaper component.
%   rtcjoin       - Computes the combination of simple streams.
%   rtcjoinstruct - Computes the combination of structured streams.
%   rtcor         - Compute the input to a component with OR activation.
%   rtcplotgpc    - Compute and plot the output of a greedy processing component.
%   rtcplotgsc    - Compute and plot the output of a greedy shaper component.

