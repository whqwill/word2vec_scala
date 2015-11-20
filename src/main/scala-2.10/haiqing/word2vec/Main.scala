package haiqing.word2vec

import javax.rmi.CORBA.Util

import org.apache.spark.{SparkContext, SparkConf}
import scala.compat.Platform.currentTime

/**
 * Created by hwang on 17.11.15.
 */
object Main {
  def main(args: Array[String]): Unit = {
    val startTime = currentTime
    val conf = new SparkConf().setAppName("Word2Vec")
    if (!conf.contains("spark.master"))
    {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    println(sc.defaultParallelism + "   " + sc.master)
    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)
    //val input = sc.textFile("raw_sentences.txt").map(line => line.split(" ").toSeq)
    val skipgram = new SkipGram().setNumPartitions(args(1).toInt).setNumIterations(args(2).toInt).setNegative(args(3).toInt)
    //val skipgram = new SkipGram().setNumPartitions(1).setNumIterations(1).setNegative(5)
    val model = skipgram.fit(input)
    val synonyms = model.findSynonyms(args(4), 10)
    //val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    println()


    //val msskipgram = new MSSkipGram(skipgram)
    val msskipgram = new MSSkipGram(skipgram).setNumPartitions(args(5).toInt).setNumIterations(args(6).toInt).setNegative(args(7).toInt).setNumSenses(args(8).toInt)

    val newModel = msskipgram.fit(input)

    for (i <- 0 to args(8).toInt-1) {
      val newSynonyms = newModel.findSynonyms(args(9)+i, 10)
      //var newSynonyms = newModel.findSynonyms("day0", 10)

      println()
      for ((synonym, cosineSimilarity) <- newSynonyms) {
        println(s"$synonym $cosineSimilarity")
      }
    }

    println("total time:"+(currentTime-startTime)/1000.0)
  }
}
