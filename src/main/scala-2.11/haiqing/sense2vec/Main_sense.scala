package haiqing.sense2vec

import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
 * Created by hwang on 09.02.16.
 */
object Main_sense {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    println(sc.defaultParallelism + "   " + sc.master)

    val input = sc.textFile(args(0),sc.defaultParallelism)

    val senseModel = new SenseAssignment().setNumRDDs(args(1).toInt).setEpoch(args(2).toInt).setMinCount(args(3).toInt).setNegative(args(4).toInt).setWindow(args(5).toInt).setVectorSize(args(6).toInt).setMultiSense(args(7).toInt).setMinCountMultiSense(args(8).toInt).setSeed(42l).setLearningRate(args(9).toFloat).setStepSize(args(10).toInt).setLocal(args(11).toBoolean)

    if (args.length == 13)
      senseModel.TrainOneSense(input,args(12))
    else
      senseModel.TrainMultiSense(input,args(12),args(13))


  }
}
