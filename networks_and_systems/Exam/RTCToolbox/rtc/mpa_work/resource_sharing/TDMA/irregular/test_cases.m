accesses = 5;
t = 3;
c = 1;
p = 1;
test1_tdma.arbiter = [0 0 1 0 0 0 0 1];
test1_tdma.time_step = 1;
wcrt1 = computeWCRTdedicatedPhase(test1_tdma, t, p, c, accesses);
if(wcrt1 ~= 24)
    error('test one failed, expected value is 24, output is %d', wcrt1);
else
    true
end
   

accesses = 11;
t = 2.5;
c = 1;
p = 1;
test2_tdma.arbiter = [1 0 0 0 0 1 0 1];
test2_tdma.time_step = 1;
wcrt2 = computeWCRTdedicatedPhase(test2_tdma, t, p, c, accesses);
if(wcrt2 ~= 32)
    error('test one failed, expected value is 32, output is %d', wcrt2);
else
    true
end


accesses = 5;
t = 11;
c = 1;
p = 1;
test3_tdma.arbiter = [0 0 1 0 0 0 0 1];
test3_tdma.time_step = 1;
wcrt3 = computeWCRTdedicatedPhase(test3_tdma, t, p, c, accesses);
if(wcrt3 ~= 32)
    error('test one failed, expected value is 32, output is %d', wcrt3);
else
    true
end


accesses = 5;
t = 19;
c = 1;
p = 1;
test4_tdma.arbiter = [0 0 1 0 0 0 0 1];
test4_tdma.time_step = 1;
wcrt4 = computeWCRTdedicatedPhase(test1_tdma, t, p, c, accesses);
if(wcrt4 ~= 40)
    error('test one failed, expected value is 40, output is %d', wcrt4);
else
    true
end
   

accesses = 11;
t = 9;
c = 1;
p = 1;
test5_tdma.arbiter = [1 0 0 0 0 1 0 1];
test5_tdma.time_step = 1;
wcrt5 = computeWCRTdedicatedPhase(test5_tdma, t, p, c, accesses);
if(wcrt5 ~= 40)
    error('test one failed, expected value is 32, output is %d', wcrt5);
else
    true
end

