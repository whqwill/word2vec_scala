package haiqing.word2vec

import org.apache.spark.{SparkContext, SparkConf}

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

    val senseModel = new SenseAssignment

    if (args(12) == "1")
      senseModel.TrainOneSense(input,args(1).toInt,args(2).toInt,args(3).toInt,args(4).toInt,args(5).toInt,args(6).toInt,args(7).toInt, args(8).toFloat, args(9).toLong, args(11))
    else if (args(12) == "2")
      senseModel.TrainTwoSenses(input,args(1).toInt,args(2).toInt,args(3).toInt,args(4).toInt,args(5).toInt,args(6).toInt,args(7).toInt, args(8).toFloat, args(9).toLong, args(10), args(11))

  }
}
