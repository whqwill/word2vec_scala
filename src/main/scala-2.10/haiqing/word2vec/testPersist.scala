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


    var input = sc.textFile("news.2013.en.short.lemma",8).map(line => line.split(" ").toSeq)

    val initialValue = new Array[Float](1000000)
    var syn0Accum = sc.accumulator(initialValue)(ArrayAccumulatorParam)
    val syn0Global = Array.fill(1000000)(1.0f)

    for(k<-1 to 100) {
      for(a<-0 to initialValue.size-1) {initialValue(a)=0.0f}
      syn0Accum = sc.accumulator(initialValue)(ArrayAccumulatorParam)
      val bcSyn0 = sc.broadcast(syn0Global)
      input = input.mapPartitions{iter=>val syn0=bcSyn0.value;val newIter=mutable.MutableList[Seq[String]]();while(iter.hasNext){newIter+=iter.next()};syn0Accum+=syn0;newIter.toIterator}.cache()
      input.count()
      println(syn0Accum.value(0))
      //for(a<-0 to syn0Global.size-1) {syn0Global(a)=syn0Accum.value(a)}
    }
  }
}

object ArrayAccumulatorParam extends AccumulatorParam[Array[Float]] {
  def zero(initialValue: Array[Float]): Array[Float] = {
    initialValue
  }
  def addInPlace(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    for(a<-0 to v1.size-1)
      v1(a) += v2(a)
    v1
  }
}