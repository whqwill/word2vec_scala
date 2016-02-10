package haiqing.word2vec

import org.apache.spark.{SparkContext, SparkConf}

import scala.compat.Platform._

/**
 * Created by hwang on 09.02.16.
 */
object Main_sense {
  def main(args: Array[String]): Unit = {
    val startTime = currentTime
    var oldTime = startTime
    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    println(sc.defaultParallelism + "   " + sc.master)

    val senseModel = new SenseAssignment

    println("load data ... ... ...")
    senseModel.loadData(sc.textFile(args(0),sc.defaultParallelism))
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("load data ... ... ...")
    senseModel.preprocessData()
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    if (args(12) == "2") {
      println("learn vocabulary without sense ... ... ...")
      senseModel.learnVocab(args(1).toInt)
      println("time:" + (currentTime - oldTime) / 1000.0)
      oldTime = currentTime
    }
    else {
      println("learn vocabulary without sense ... ... ...")
      senseModel.learnVocabWithoutSense(args(1).toInt)
      println("time:" + (currentTime - oldTime) / 1000.0)
      oldTime = currentTime
    }

    println("make sentences ... ... ...")
    senseModel.makeSentences(args(2).toInt)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("split RDD ... ... ...")
    senseModel.splitRDD(args(3).toInt)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    if (args(12) == "2") {
      println("initialize parameters ... ... ...")
      senseModel.initializeParameters(args(10), args(4).toInt)
      println("time:" + (currentTime - oldTime) / 1000.0)
      oldTime = currentTime
    }
    else {
      println("initialize parameters ... ... ...")
      senseModel.initializeParameters(args(4).toInt)
      println("time:" + (currentTime - oldTime) / 1000.0)
      oldTime = currentTime
    }

    println("train (local version) ... ... ...")
    senseModel.train_local(args(5).toInt,args(6).toInt,args(7).toInt,args(8).toFloat,args(9).toLong)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    if (args(12) == "2") {
      println("wrote to file ... ... ...")
      senseModel.writeToFile(args(11))
      println("time:" + (currentTime - oldTime) / 1000.0)
      oldTime = currentTime
    }
    else {
      println("wrote to file ... ... ...")
      senseModel.writeToFile(args(10))
      println("time:" + (currentTime - oldTime) / 1000.0)
      oldTime = currentTime
    }

    println("total time:"+(currentTime-startTime)/1000.0)
  }
}
