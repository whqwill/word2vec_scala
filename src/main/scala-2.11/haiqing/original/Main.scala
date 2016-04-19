package haiqing.original

import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by hwang on 19.04.16.
 */
object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    println(sc.defaultParallelism + "   " + sc.master)

    val input = sc.textFile(args(0),sc.defaultParallelism).map(line => line.split(" ").toSeq)

    val word2vec = new Word2Vec().setNumPartitions(sc.defaultParallelism).setNumIterations(args(1).toInt).setMinCount(args(2).toInt).setWindowSize(args(3).toInt).setVectorSize(args(4).toInt).setLearningRate(args(5).toDouble).setSeed(42L)

    val model = word2vec.fit(input)

    println("day")
    var synonyms = model.findSynonyms("day", 30)
    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    /*
    println("apple")
    synonyms = model.findSynonyms("apple", 30)
    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    println("bank")
    synonyms = model.findSynonyms("bank", 30)
    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }
    */

  }
}
