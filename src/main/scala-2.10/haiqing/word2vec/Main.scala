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
    val words = input.flatMap(x => x).filter(Preprocessing.filter1).filter(Preprocessing.filter2).map(Preprocessing.map1)

    val skipgram = new SkipGram().setNumPartitions(args(1).toInt).setNumIterations(args(2).toInt).setNegative(args(3).toInt)

    val model = skipgram.fit(words)
    val synonyms = model.findSynonyms(args(4), 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    println()

    //skipgram.cleanSyn()

    val msskipgram = new MSSkipGram(skipgram).setNumPartitions(args(5).toInt).setNumIterations(args(6).toInt).setNegative(args(7).toInt).setNumSenses(args(8).toInt)

    val newModel = msskipgram.fit(words)

    for (i <- 0 to args(8).toInt-1) {
      val newSynonyms = newModel.findSynonyms(args(9)+i, 10)

      println()
      for ((synonym, cosineSimilarity) <- newSynonyms) {
        println(s"$synonym $cosineSimilarity")
      }
    }

    println("total time:"+(currentTime-startTime)/1000.0)
  }
}

object Preprocessing {
  def map1(s: String): String={
    var i = 0
    while (i < s.length && s(i) != ''') {i+=1}
    if (i == s.length)
      s.toLowerCase.filter((c: Char) => c >= 'a' && c <= 'z')
    else
      s.substring(0,i).toLowerCase.filter((c: Char) => c >= 'a' && c <= 'z')
  }

  def filter1(s: String): Boolean={
    if (s.length == 0)
      false
    else
      true
  }

  def filter2(s: String): Boolean={
    if (s(0) >= '0' && s(0) <= '9')
      false
    else
      true
  }
}