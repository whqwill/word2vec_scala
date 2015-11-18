package haiqing.word2vec

import javax.rmi.CORBA.Util

import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by hwang on 17.11.15.
 */
object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ss").setMaster("local")
    val sc = new SparkContext(conf)
    val input = sc.textFile("raw_sentences.txt").map(line => line.split(" ").toSeq)
    val word2vec = new Word2Vec().setNegative(15)
    val model = word2vec.fit(input)
    val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }
  }
}
