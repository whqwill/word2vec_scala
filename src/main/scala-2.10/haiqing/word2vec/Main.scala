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
    val skipgram = new SkipGram().setNumPartitions(args(1).toInt).setNumIterations(args(2).toInt).setNegative(args(3).toInt)
    //val word2vec = new Word2Vec().setNumPartitions(4).setNumIterations(2).setNegative(5)
    val model = skipgram.fit(input)
    val synonyms = model.findSynonyms(args(4), 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }
    println("total time:"+(currentTime-startTime)/1000.0)
    val msskipgram = new MSSkipGram()
    msskipgram.initFromSkipGram(skipgram)

    val newModel = msskipgram.fit(input)
    val newSynonyms = newModel.findSynonyms(args(5), 10)

    for((synonym, cosineSimilarity) <- newSynonyms) {
      println(s"$synonym $cosineSimilarity")
    }
  }
}
