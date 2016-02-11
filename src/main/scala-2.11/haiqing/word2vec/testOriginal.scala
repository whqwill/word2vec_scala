package haiqing.word2vec

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.{SparkContext, SparkConf}

import scala.compat.Platform._

/**
 * Created by hwang on 11.02.16.
 */
object testOriginal {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    println(sc.defaultParallelism + "   " + sc.master)

    if (args(0) == "1")
      println(sc.textFile(args(1)).count())
    /*
    val in = sc.textFile(args(0),sc.defaultParallelism)

    val num = in.count()
    println(num)

    val numRDD = num.toInt/(2500*sc.defaultParallelism)

    val split = in.cache().randomSplit(new Array[Double](numRDD).map(x => x + 1))
    println("split.size="+split.size)
    println("numRDD="+numRDD)
    //for (k <- 1 to 10) {
      //split(k).cache()
    //}
    for (i <- 1 to 100) {
      val startTime = currentTime
        //println("iteration " + k)

        //in.sample(false,1.0f/numRDD).count()
        //in.count()
        //split(k%numRDD).foreachPartition { iter =>
        //in.sample(false,1.0f/numRDD).foreachPartition { iter =>
        //var count = 0
        //for (sentence <- iter) {
        // count += sentence.size
        //}
        //}

      in.foreachPartition { iter =>
        var count = 0
        for (sentence <- iter) {
          count += sentence.size
        }
      }


      println("total time:" + (currentTime - startTime) / 1000.0)
    }
*/
return
    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)

    val word2vec = new Word2Vec().setNumPartitions(sc.defaultParallelism).setNumIterations(1)

    val startTime = currentTime
    val model = word2vec.fit(input)
    println("total time:" + (currentTime - startTime) / 1000.0)

    val synonyms = model.findSynonyms("bank", 40)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

  }
}
