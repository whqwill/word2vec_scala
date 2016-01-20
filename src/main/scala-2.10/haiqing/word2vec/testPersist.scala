package haiqing.word2vec

import org.apache.spark.rdd.RDD
import org.apache.spark.{AccumulatorParam, SparkContext, SparkConf}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder
import scala.compat.Platform._

/**
 * Created by hwang on 15.01.16.
 */
object testPersist {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("testPersist")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    println(sc.defaultParallelism + "   " + sc.master)

    


    //var input = sc.textFile("news.2013.en.short.lemma",8).map(line => line.split(" ").toSeq).randomSplit(new Array[Double](10).map(x=>x+1))

    ///val initialValue = new Array[Float](1000000)
    //var syn0Accum = sc.accumulator(initialValue)(ArrayAccumulatorParam)
    //val syn0Global = Array.fill(1000000)(1.0f)
    //var tmpRDD = input(0)

    //scala.Vector

/*
    for(k<-1 to 1000) {
      println("iteration "+k)
      for(a<-0 to initialValue.size-1) {initialValue(a)=0.0f}
      syn0Accum = sc.accumulator(initialValue)(ArrayAccumulatorParam)
      val bcSyn0 = sc.broadcast(syn0Global)
      //tmpRDD = input((k-1)%10).mapPartitions{iter=>val syn0=bcSyn0.value;val newIter=mutable.MutableList[Seq[String]]();while(iter.hasNext){newIter+=iter.next()};syn0Accum+=syn0;newIter.toIterator}.cache()
      //tmpRDD = input.mapPartitions{iter=>val syn0=bcSyn0.value;val newIter=mutable.MutableList[Seq[String]]();while(iter.hasNext){newIter+=iter.next()};syn0Accum+=syn0;newIter.toIterator}.cache()
      tmpRDD = input((k-1)%10).mapPartitions{iter=>val syn0=bcSyn0.value;val newIter=mutable.MutableList[Seq[String]]();while(iter.hasNext){newIter+=iter.next()};syn0Accum+=syn0;newIter.toIterator}.cache()
      //tmpRDD = input.mapPartitions{iter=>iter}.cache()
      tmpRDD.count()
      input((k-1)%10).unpersist()
      //input.unpersist()
      //input = tmpRDD
      input((k-1)%10) = tmpRDD
      println(syn0Accum.value(0))
      //for(a<-0 to syn0Global.size-1) {syn0Global(a)=syn0Accum.value(a)}
    }*/
  }
}
