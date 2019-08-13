# 并行计算和分布式一些术语和框架

最近看了ps-lite，了解了一下第三代ps的架构，然而在文章里面对于分布式的描述很多，并且结合自己认真，发现一些概念，术语常常让人看得眼花缭乱，下面是对分布式或者并行计算的一些浅浅的总结，有不对的地方请指正出来，后续会修改。

有以下几个问题在看的时候不解：

1.OpenMP，MPI，OpenMPI，NCCL等等傻傻分不清楚？

而分布式往往涉及到的并行计算，跟之间了解过的tensorflow的mkl-dnn库(底层用OpenMP实现)，以及MPI的框架的实现OpenMPI又是什么？

2.大数据分布式框架，还有深度学习中用的分布式到底有什么不同？

大数据领域经常提起的是hadoop/spark的mapreduce框架，结合之前horovod分布式用到了的MPI框架，NCCL框架，还有ring-allreduce框架，以及ps框架，他们之间到底有什么不同和差异，在什么场景下使用？

3.ps架构发展历史，不同公司的实现，第三代ps-lite的原理？

由于协议的不同，从同步并行(BSP，Bulk Synchronous Parallel)到异步分布式（AP，Asynchronous Parallel），还有介于两者中间(SSP，Stale Synchronous Parallel)各自的优点缺点，第三代ps-lite有什么神奇地方？
### 针对第1个问题：

MPI(Message Passing Interface)，从字面理解就是消息传递库，一些网络上的MPI说法其实都是错的，简单来说，MPI定义了一种标准，美国并行计算中心工作会议在92年定的，如今发展到MPI-2版本，而OpenMPI，MPICH，LAM是MPI的实现[1]，他们的关系取个简单例子：

![openmpi](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/openmpi.png?raw=true "openmpi")![mpich](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/mpich.png?raw=true "mpich")![lam](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/lam.png?raw=true "lam")

如NBI篮球联赛，定义篮球标准(MPI)，一个圆形，能充气，并且体积不超过10 * 10 *10，而针对这种标准，不同公司为了成为官方的指定球类品牌，生成了许多这种标准篮球，并打上自己标志，如耐克，李宁，魔腾（OpenMPI,MPICH,LAM）。

1.MPI是一个库，而不是一门语言；

2.MPI是一种标准或者规范的代表，而不特指某一个对它的具体实现；

3.MPI是一种消息传递编程模型，并成为这种编程模型的代表和事实上的标准；

以一个mpichi简单例子说明[2]：

```
sudo apt-get install libcr-dev mpich2 mpich2-doc
/* C Example */
#include <mpi.h>
#include <stdio.h>
 
int main (int argc, char* argv[])
{
 int rank, size;
 
 MPI_Init (&argc, &argv);   /* starts MPI */
 MPI_Comm_rank (MPI_COMM_WORLD, &rank);    /* get current process id */
 MPI_Comm_size (MPI_COMM_WORLD, &size);    /* get number of processes */
 printf( "Hello world from process %d of %d\n", rank, size );
 MPI_Finalize();
 return 0;
}
编译运行及显示结果
mpicc mpi_hello.c -o hello
mpirun -np 2 ./hello
Hello world from process 0 of 2
Hello world from process 1 of 2
```
可以看到启动两个进程szie=1，每个进程分别获取了自己的rank，0和1编号，可以看出。

![openmp](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/openmpLogo.200pix.gif?raw=true "openmp")
OpenMP(Open Multi-Processing)，可以用来显式地指导多线程、共享内存式程序的并行化，由三个主要的API组件构成，编译器指令，运行库，环境变量[3]。OpenMP的内存模型是统一/共享的内存模型(unified/shared memory)，比如我的电脑中有8个核心，但是只有一块内存，各个核心通过内存来分享交换数据。它和MPI的区别在于，MPI不仅可以用于unified/shared memory，而且可以distribute memory，在一个集群内有好几台服务器，每一台有各自的内存和CPU，需要通过以太网来交换数据。

也是以一个简单例子来说明OpenMP[4]：

```
#include <omp.h>
#include <iostream>
#include <time.h>
using namespace std;
int main(){
	clock_t t_start=clock();
	# pragma omp parallel for num_threads(2)
	for(int i=0;i<2000000000;i++){
	}
	time_t t_end = clock();
	cout<<"Run time: "<<(double)(t_end - t_start) / CLOCKS_PER_SEC<<"S"<<endl;
	return 0;
}
```
一个C语言的程序，统计i=0到2000000000耗时，在不加#pragma，和加#pragma，运行程序的时间是不同的，不加#pragma大概是4秒，加了大概是2秒，所以只要在程序适当位置#pragma就会多线程并行执行下面内容，常见是后面加for语句，就会自动多线程累加，当然前提是for保证每次循环之间无数据相关性，这边是设置num_thread=2,会启动两个线程进行累加，也可以下面加omp_set_num_threads()库函数的使用，或者环境变量OMP_NUM_THREADS的设置，默认是CPU的数量或者内核数量，优先级:字句num_thread>omp_set_num_threads()库函>环境变量OMP_NUM_THREADS>默认，其中tensorflow+mkl-dnn就用到了openmp，针对于cpu优化，其主要目的是在运行过程中对计算并行加速。
深度学习框架tensorflow带mkl-dnn库有相应的环境变量跟这个类似，由于mkl-dnn底层是用Openmp写的，所以也有这个环境变量。
这边可以看出MPI和OpenMP的区别：
1.从场景上看：MPI场景可以针对单机/多机使用的，进程级别并行，openmp是场景是单机使用的，是线程式级别的运行，单机下上面的opemp的代码如果是算0-2000000000，既可以openmp开2个线程累加，也可以用MPI开2个进程累加，但是MPI2个进程之间还涉及到进程间的通信(IPC)，通过消息机制，分配每个进程的任务，比如让第一个进程算0-1000000000，第二个进程算1000000001-2000000000，最终每个进程各自算完，再调用MPI_Allreduce函数把每个进程数据处理后再分发回每个进程。
2.单机下，从代码上看openmp貌似比较简洁，只需要加一些制导语句即可，不用大幅修改原来的C代码，而MPI需要自己配置每个进程相应的任务。
3.单机下，从使用内存上看，MPI由于每个进程独立一块内存空间，所以内存使用肯定比openmp来的大，但是在单机numa下，MPI编程模型天生强制比较好的data locality(内部做了很多内存优化)，所以数据内存都在同一个numa单元下，而 而openmp由于开的线程，除非做了affinity，不然新开的线程是随机CPU的，没法保证保证操作的内存都属于同一个numa单元下，所以单机下MPI的效率不输openmp，甚至还要更好[5]。
所以在多机多核情况当然可以两者结合，即MPI+OpenMP。

再讲到NCCL，英伟达的通信库，NCCL(NVIDIA Collective Communication Library)[6]
Multi-GPU and multi-node collective communication primitives，看这个解释来说针对多节点，多GPU的使用场景，而上面所说的MPI和openMP都是针对在CPU架构讨论的，而NCCL和这些的不同点在于使用GPU而言，对于深度学习更多的是GPU的并行的计算，所以NCCL有点类似于MPI[3]，有点类似说在MPI基础+CUDA的味道，我想英伟达应该是借鉴这两种并行计算思想，在多机多卡上实现了通信，实现多机多卡的并行计算，在官网上看它的特性是支持多线程和多进程应用程序，没有做深入的探究，详细可以看官网或者其他人的解释[7]。
[1]https://wenku.baidu.com/view/a6ae37fbc8d376eeaeaa31a7.html
[2]https://www.jb51.net/article/74648.htm
[3]https://blog.csdn.net/magicbean2/article/details/75530667
[4]https://blog.csdn.net/yongh701/article/details/51351692
[5]https://www.zhihu.com/question/20188244
[6]https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/mpi.html
[7]https://www.leiphone.com/news/201710/WiIOgnsK3cuq6tjy.html?uniqueCode=sVjAZ8qRteHbFfJb

### 针对第2个问题：
在看第二个问题，在数据领域，hadoop集群或者spark提到的mapreduce，其实是一种编程方式，每个公司提供的API有着很大的区别，这个跟MPI不同，MPI是有同一标准的，每个公司实现都是按照这个标准来，而mapreduce只是一种编程的方式，在实现时，在Map函数中指定对各分块数据的处理过程，在Reduce函数中指定如何对分块数据处理的中间结果进行归约。用户只需要指定Map和Reduce函数来编写分布式的并行程序，不需要关心如何将输入的数据分块、分配和调度，同时系统还将处理集群内节点失败及节点间通信的管理等。而MPI仅仅是一个并行计算标准，没有相应的分布式文件系统的支撑，在大数据场景下大文件的存储及访问都会成为一个问题，同时用户还需要考虑集群节点之间的通信协调、容错等问题，这些使得MPI的编程难度比较大，集群本身的规模也很难做到像MapReduce那样的超大规模。
看一下Mapreduce的框架：

![mapreduce](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/mapreduce.jpg?raw=true "mapreduce")

mapreduce包含了map函数和reduce函数，map函数和reduce函数是交给用户实现的，这两个函数定义了任务本身。
map函数：接受一个键值对（key-value pair），产生一组中间键值对。MapReduce框架会将map函数产生的中间键值对里键相同的值传递给一个reduce函数。reduce函数：接受一个键key，以及相关的一组值（value list），将这组值进行合并产生一组规模更小的值（通常只有一个或零个值）。
举个简单例子：想统计下过去10年计算机论文出现最多的几个单词，看看大家都在研究些什么，那收集好论文后，该怎么办呢？[1] 
这个时候看一下mapreduce框架如何处理：
统计词频的MapReduce函数的核心代码非常简短，主要就是实现这两个函数。
```
map(String key, String value):
        // key: document name
        // value: document contents 
        for each word w in value: 
                EmitIntermediate(w, "1");
```
```
reduce(String key, Iterator values): 
     // key: a word 
     // values: a list of counts 
     int result = 0; 
     for each v in values:
         result += ParseInt(v); 
     Emit(AsString(result)); 
```
在统计词频的例子里，map函数接受的键（key）是文件名，值（value）是文件的内容，map逐个遍历单词，每遇到一个单词word就产生一个中间键值对<w, "1">（表示单词w咱又找到了一个）；MapReduce将键相同（都是单词w）的键值对传给reduce函数，这样reduce函数接受的键就是单词w，值是一串"1"（最基本的实现是这样，但可以优化），个数等于键为w的键值对的个数，然后将这些“1”累加就得到单词w的出现次数。最后这些单词的出现次数会被写到用户定义的位置，存储在底层的分布式存储系统（GFS或HDFS）。
一切都是从最上方的user program开始的，user program链接了MapReduce库，实现了最基本的Map函数和Reduce函数。图中执行的顺序都用数字标记了。
（1）MapReduce库先把user program的输入文件划分为M份（M为用户定义），每一份通常有16MB到64MB，如图左方所示分成了split0~split4；然后使用fork将用户进程拷贝到集群内其它机器上。

（2）user program的副本中有一个称为master，其余称为worker，master是负责调度的，为空闲worker分配作业（Map作业或者Reduce作业），worker的数量也是可以由用户指定的。

（3）被分配了Map作业的worker，开始读取对应分片的输入数据，Map作业数量是由M决定的，和split一一对应；Map作业从输入数据中抽取出键值对，每一个键值对都作为参数传递给map函数，map函数产生的中间键值对被缓存在内存中（环形缓冲区kvBuffer）。

（4）缓存的中间键值对会被定期写入本地磁盘（spill），而且被分为R个区，R的大小是由用户定义的，将来每个区会对应一个Reduce作业；这些中间键值对的位置会被通报给master，master负责将信息转发给Reduce worker。

（5）master通知分配了Reduce作业的worker它负责的分区在什么位置（肯定不止一个地方，每个Map作业产生的中间键值对都可能映射到所有R个不同分区），当Reduce worker把所有它负责的中间键值对都读过来后，先对它们进行排序，使得相同键的键值对聚集在一起。因为不同的键可能会映射到同一个分区也就是同一个Reduce作业，所以排序是必须的。

（6）reduce worker遍历排序后的中间键值对，对于每个唯一的键，都将键与关联的值传递给reduce函数，reduce函数产生的输出会添加到这个分区的输出文件中。

（7）当所有的Map和Reduce作业都完成了，master唤醒正版的user program，MapReduce函数调用返回user program的代码。

反过头来看如果用MPI来实现这样的一个流程，可不可以，当然也可以，MPI也可以起多个进程，每个进程内只统计部分的文章，然后每个进程也可以生成key-value形式，产生单词-单词数量的，然后最后相同key的累加，这样的做法也是可以的，但是MPI起多个进程，如果一个进程挂掉就会导致整个MPI挂掉，而mapreduce基于整个hadoop生态圈中，某个进程挂掉容错是很高，有故障恢复功能，并且MPI的这样方式，对于这么大量论文该如何读取，如果每台机器都保存完整数据，然后根据进程rank号来读取相应编号的论文，那么每台机器存储大量论文又是一个难点，如果所有论文都保存在同一台机器，以挂载方式挂载到每台MPI的机器上，那么大量访问也是个难点，并且MPI完整程序需要人为的编写，而不像mapreduce一样只关心和业务相关的函数map和reduce。

再来看一下ps和ring-allreduce用于深度学习分布式，其实ps也有一种架构是tree-allreduce，allreduce这个词，其实是mpi的原语，在前面有说道MPI有个函数就是allreduce，它的作用就是把每个子集的算的局部统计量整合成全部统计量，然后再分配给每个子集。有点类似于mapreduce的味道，但是还是不一样，**重点在于allreduce有再分配会各个子集的这样的流程**。
前面mapreduce是一种面向通用任务处理的多阶段执行任务的方式，如上面的例子所示，map函数接受一对键值对，产生中间键值对，然后reduce函数会根据中间键值对，对相同键聚集在一起。
而allreduce，举个例子，深度学习中一般传递是模型的参数，训练开始时，假设每台机器是不同的数据，犹如map函数一样，key是每台机器分配到的训练数据名称，value是训练数据内容，一轮迭代后，每台机器(或者每张GPU)产生相应模型参数，也可以认为产生了中间键，key是每个参数名称（w0...wn）,value是每个参数值，而把每台机器(或者每张GPU)相同的key的参数融合后，又返回给每台机器(或者每张GPU)继续下一轮迭代。所以从流程看，allreduce是适合机器学习这种分布式的抽象。

机器学习算法和计算机领域的其他算法相比，有自己的一些独特特点。例如：迭代性，模型的更新并非一次完成，需要循环迭代多次；容错性，即使在每个循环中产生一些错误，模型最终的收敛不受影响；参数收敛的非均匀性，模型中有些参数经过几个循环便不再改变，其他参数需要很长时间收敛。这些特点决定了机器学习系统的设计和其他计算系统的设计有很大不同，因此理想中的分布式机器学习任务，并不能随着机器的增加而能力线性提升，因为大量资源都会浪费在通讯，等待，协调，这些时间可能会占据大部分比例。
PS架构分为Worker和Server的角色，如果左边是worker角色，右边是server角色，**在同步情况**下，明显上面ps架构明显可以看出server的GPU0会是性能的瓶颈，是针对多机来说，因为多机情况下进程间是通过socket来通信的，这边的性能的瓶颈分为两种：
（1）因为同步情况下由于每台机器性能不一致，其他机器worker到达server时会等待最慢的worker到达。
（2）在大规模的worker数量下，server侧带宽也会成为瓶颈。
要解决多机的这两个问题，ps架构可以引入多个server，来减少单个server的带宽负担，以异步方式来减少等待。
所以ps-lite第三代ps架构就是解决这两个问题，当然最新的byte-ps也是解决这两个问题，并且更优秀，原因在于ps-lite的worker和server还是属于同一台机器内，而byte-ps是把worker和ps一一对应起来，充分利用每台机器的带宽。
ring-allreduce也是用来解决带宽问题，**最大程度利用带宽**提出的一种分布式架构[2]，详细可以看卓龙对ring-allreduce的介，或者搜索ring-allreduce会有相应的ring-allreduce的介绍，这里不再赘述，而本身ring-allreduce是同步型的，是没有解决等待问题，还是会受限于最慢的那个GPU。
而像RMDA技术其实是加速了处理速度，提高了吞吐量。

所以在深度学习分布式下，要解决的问题有，等待，带宽，处理能力，协调，不同的架构和硬件推陈出新都是为了解决这些问题。

在额外说一个问题，horovod的框架中的stack是这样的：



[1]https://blog.csdn.net/Little_Fire/article/details/80605233

[2]<https://mp.weixin.qq.com/s?__biz=MzAwNDU4MjIyOA==&mid=401609990&idx=1&sn=843354795c8ed298f1b9aedc0539be7b&mpshare=1&scene=1&srcid=1205DNviVc5rvuILTFgisfAO#rd>

[3]https://www.jianshu.com/p/8c0e7edbefb9

## 针对第3个问题：
ps架构来并行学习，一开始也是面临着很多挑战，如我上面说所说的，由于集群本身的Unequal performance machines和网络通信上的Low bandwidth, High delay问题。集群越大，线性扩展的代价就越大，网络通信会占据时间开销的主要部分

![挑战](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/1.png?raw=true "挑战")
最早的PS是采用BSP协议方式：

![BSP](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/3.png?raw=true "BSP")

接下来介绍一下ps-lite，是李沐设计的参数服务器[1],结合了前人经验，异步型并且借鉴SSP思想（只不过把迭代改成时间），也参照了一致性hash分片存储的思想来存模型参数，新增节点不需要重新运行系统，故障节点也不会导致中断计算，有关于一致性hash算法后面会详细提到。
ps-lite目前在DMLC（Distributed (Deep) Machine）项目中处于核心基础地位，因为大部分的分布式机器学习算法都会基于它来进行，包括DMLC最新推出的热门深度学习框架MXNet。ps-lite的代码整体非常简单，便于修改和移植，而且DMLC项目组目前也给它增加了资源管理器的集成，使得Yarn能够来管理参数服务器的资源分配。
![ps-lite](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/11.png?raw=true "ps-lite")
![ps-lite](https://github.com/wangjm12138/mxnet_summarize/blob/master/markdown_pic/12.png?raw=true "ps-lite")

[1]https://github.com/dmlc/ps-lite
[2]http://www.learn4master.com/machine-learning/parameter-server/parameter-server-resources
\