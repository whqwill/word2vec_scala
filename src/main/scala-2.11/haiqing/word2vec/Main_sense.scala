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

    val senseModel = new SenseAssignment().setNumRDDs(args(1).toInt).setEpoch(args(2).toInt).setMinCount(args(3).toInt).setNegative(args(4).toInt).setWindow(args(5).toInt).setVectorSize(args(6).toInt).setSeed(42l).setLocal(true)


    if (args.length == 8)
      senseModel.TrainOneSense(input,args(7))
    else
      senseModel.TrainTwoSenses(input,1000,args(7),args(8))

  }
}
