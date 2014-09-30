Project-2
=========

A Study in Parallel Algorithms : Stream Compaction

(For part 2 and 3 questions) 
# Scan Comparison
![](https://raw.githubusercontent.com/gouzhen1/Project2-StreamCompaction/blob/master/scanChart.bmp)
At a first glance, the naive implementation of scan performs worse than the serial version for all number of N.
This is probably because the parallel algorithm used has a complexity of O(N*Log(N)) where the serial version has O(N)
However when utilizing shared memory, the GPU version gradually catches up the serial version as n gets bigger
and out perform it at around n= 5,000,000, the use of shared memory clearly speeds it up and it runs the same
 algorithm which has O(N*log(N)) this is probably why the GPU catches up when n gets largers as the log(N) term slows down

Part 4 
# Stream Compaction Comparison
![](https://raw.githubusercontent.com/gouzhen1/Project2-StreamCompaction/blob/master/streamCompactCompare.bmp)
Both my GPU implementation and Thrust's beat the serial version no matter how big n was and as n gets larger the bigger advantage.
Mine is slower than Thrust because my scan doesn't use the work efficient algorithm and doesn't solve bank conflicts. And this is 
where to improve and boost my implementation's performance.

References
http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html