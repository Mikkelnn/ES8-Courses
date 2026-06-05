function [upperCurve lowerCurve] = computeCurves(tuples, upperCurve, lowerCurve, distance)
%[upperCurve lowerCurve] = computeCurves(tuples, upperCurve, lowerCurve, distance)
%
%This function updates a give set of arrival curves with data stored
%in the data structure tuples. Tuples is a structure containing
%informations about the min/max number of cache misses for the worst/best
%case execution time.
%
%upperCurve ... in/out ... the updated upper arrival curve
%lowerCurve ... in/out ... the updated lower arrival curve
%tuples     ... in     ... the data structure containing the cache-miss
%                          pattern
%distance   ... in     ... the number of subsequent super-block considered
%                          in this iteration
%
%See also computeTuples, arrivalCurvesTuples


    number_of_tuples = length(tuples.best_et_u) - distance;

    for tuples_counter = 1:number_of_tuples
        
        %compute the discreet points, for which cache-miss values have to
        %be approximated. This can be in both directions (descending,
        %ascending).
        
        %a) for the upper execution times
        if(tuples.best_et_u(tuples_counter) <=tuples.worst_et_u(tuples_counter))
            approximation_array_u = tuples.best_et_u(tuples_counter):tuples.worst_et_u(tuples_counter);
        else
            approximation_array_u = tuples.best_et_u(tuples_counter):-1:tuples.worst_et_u(tuples_counter);
        end
        approximation_array_u = approximation_array_u + 1;
        
        %b) for the lower execution times
        if(tuples.best_et_l(tuples_counter) <=tuples.worst_et_l(tuples_counter))
            approximation_array_l = tuples.best_et_l(tuples_counter):tuples.worst_et_l(tuples_counter);
        else
            approximation_array_l = tuples.best_et_l(tuples_counter):-1:tuples.worst_et_l(tuples_counter);
        end
        approximation_array_l = approximation_array_l + 1;
        
        
        
        %compute the approximation between the best case point and the
        %worst case point for one specific sequence (start and distance)
        
        %a) for the upper execution times
        for approximation_counter = 1:length(approximation_array_u)
            %ceil, for communication time != integers
            current_index = ceil(approximation_array_u(approximation_counter));
           
            
           %special case: wc = bc (e.g. in the first iteration, where
           %\delta = 0.
           if (tuples.worst_et_u(tuples_counter) - tuples.best_et_u(tuples_counter) == 0)   
             if (upperCurve(current_index) <  tuples.misses_wc_max(tuples_counter))
                upperCurve(current_index) =  tuples.misses_wc_max(tuples_counter);
             end
             if (upperCurve(current_index) <  tuples.misses_bc_max(tuples_counter))
                upperCurve(current_index) =  tuples.misses_bc_max(tuples_counter);
             end
             
             if (upperCurve(current_index) < tuples.misses_wc_min(tuples_counter))
                upperCurve(current_index) = tuples.misses_wc_min(tuples_counter);  
             end
             
             if (upperCurve(current_index) < tuples.misses_bc_min(tuples_counter))
                upperCurve(current_index) = tuples.misses_bc_min(tuples_counter);  
             end
           
             if (lowerCurve(current_index) > tuples.misses_bc_min(tuples_counter))
                lowerCurve(current_index) = tuples.misses_bc_min(tuples_counter);  
             end
             if (lowerCurve(current_index) > tuples.misses_wc_min(tuples_counter))
                lowerCurve(current_index) = tuples.misses_wc_min(tuples_counter);  
             end
             
           %normal case, when WC > BC  
           else
               
               %the approximation is done between the tuples that represent
               %the maximum and minimal number of cahce-misses for a given
               %sequence of superblocks. 
                           
               misses_upper = tuples.misses_bc_max(tuples_counter) ...
                   + (current_index-1 - tuples.best_et_u(tuples_counter)) * ...
                   ( (tuples.misses_wc_max(tuples_counter) - tuples.misses_bc_max(tuples_counter)) / (tuples.worst_et_u(tuples_counter) - tuples.best_et_u(tuples_counter)));
               misses_lower = tuples.misses_bc_min(tuples_counter) ...
                   + (current_index-1 - tuples.best_et_u(tuples_counter)) * ...
                   ( (tuples.misses_wc_min(tuples_counter) - tuples.misses_bc_min(tuples_counter)) / (tuples.worst_et_u(tuples_counter) - tuples.best_et_u(tuples_counter)));

               %update the curve for those values
               if (upperCurve(current_index) < misses_upper)
                   upperCurve(current_index) = misses_upper;
               end
               
               if (upperCurve(current_index) < misses_lower)
                   upperCurve(current_index) = misses_lower;
               end

               if (lowerCurve(current_index) > misses_lower)
                   lowerCurve(current_index) = misses_lower;

               end
           end     
        end
        
        %b) for the lower execution times
        for approximation_counter = 1:length(approximation_array_l)
            current_index = ceil(approximation_array_l(approximation_counter));
           
            
           %special case: wc = bc (e.g. in the first iteration, where
           %\delta = 0.
           if (tuples.worst_et_l(tuples_counter) - tuples.best_et_l(tuples_counter) == 0)   
             if (upperCurve(current_index) <  tuples.misses_wc_max(tuples_counter))
                upperCurve(current_index) =  tuples.misses_wc_max(tuples_counter);
             end
             if (upperCurve(current_index) <  tuples.misses_bc_max(tuples_counter))
                upperCurve(current_index) =  tuples.misses_bc_max(tuples_counter);
             end
             if (upperCurve(current_index) < tuples.misses_bc_min(tuples_counter))
                upperCurve(current_index) = tuples.misses_bc_min(tuples_counter);  
             end
             if (upperCurve(current_index) < tuples.misses_wc_min(tuples_counter))
                upperCurve(current_index) = tuples.misses_wc_min(tuples_counter);  
             end
           
             if (lowerCurve(current_index) > tuples.misses_bc_min(tuples_counter))
                lowerCurve(current_index) = tuples.misses_bc_min(tuples_counter);  
             end
             if (lowerCurve(current_index) > tuples.misses_wc_min(tuples_counter))
                lowerCurve(current_index) = tuples.misses_wc_min(tuples_counter);  
             end
             
           %normal case, when WC > BC  
           else
               
               %the approximation is done between the tuples that represent
               %the maximum and minimal number of cahce-misses for a given
               %sequence of superblocks. 
                           
               misses_upper = tuples.misses_bc_max(tuples_counter) ...
                   + (current_index-1 - tuples.best_et_l(tuples_counter)) * ...
                   ( (tuples.misses_wc_max(tuples_counter) - tuples.misses_bc_max(tuples_counter)) / (tuples.worst_et_l(tuples_counter) - tuples.best_et_l(tuples_counter)));
               misses_lower = tuples.misses_bc_min(tuples_counter) ...
                   + (current_index-1 - tuples.best_et_u(tuples_counter)) * ...
                   ( (tuples.misses_wc_min(tuples_counter) - tuples.misses_bc_min(tuples_counter)) / (tuples.worst_et_l(tuples_counter) - tuples.best_et_l(tuples_counter)));

               %update the curve for those values
               if (upperCurve(current_index) < misses_upper)
                   upperCurve(current_index) = misses_upper;
               end
               
               if (upperCurve(current_index) < misses_lower)
                   upperCurve(current_index) = misses_lower;
               end
               

               if (lowerCurve(current_index) > misses_lower)
                   lowerCurve(current_index) = misses_lower;

               end
           end     
        end
    end

end