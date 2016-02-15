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

    val senseModel = new SenseAssignment().setNumRDDs(args(1).toInt).setIterations(args(2).toInt).setLocal(true).setMinCount(args(3).toInt)

    //senseModel.TrainOneSense(input,args(4))

    senseModel.TrainTwoSenses(input,args(4),args(5))

  }
}
