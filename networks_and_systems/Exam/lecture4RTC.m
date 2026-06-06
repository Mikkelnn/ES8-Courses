for N = 1:8
    r = 8*(100000 + 100000/N);
    burst = 8*(100*N + 100);

    alpha{N} = rtccurve([0,0,r]);
end
rtcplot(alpha{:})