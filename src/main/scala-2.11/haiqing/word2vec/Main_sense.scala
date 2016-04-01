package haiqing.word2vec

import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable
import scala.compat.Platform._

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

    val senseModel = new SenseAssignment().setNumRDDs(args(1).toInt).setEpoch(args(2).toInt).setMinCount(args(3).toInt).setNegative(args(4).toInt).setWindow(args(5).toInt).setVectorSize(args(6).toInt).setMultiSense(args(7).toInt).setMinCountMultiSense(args(8).toInt).setSeed(42l).setLocal(args(9).toBoolean)

    if (args.length == 11)
      senseModel.TrainOneSense(input,args(10))
    else
      senseModel.TrainMultiSense(input,args(10),args(11))
  }
}
