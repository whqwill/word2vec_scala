package haiqing.word2vec

import javax.rmi.CORBA.Util

import org.apache.spark.{SparkContext, SparkConf}
import scala.compat.Platform.currentTime

/**
 * Created by hwang on 17.11.15.
 */
object Main_local {
  def main(args: Array[String]): Unit = {
    val startTime = currentTime
    val conf = new SparkConf().setAppName("Word2Vec")
    if (!conf.contains("spark.master"))
    {
      conf.setMaster("local[4]")
    }
    val sc = new SparkContext(conf)



    println(sc.defaultParallelism + "   " + sc.master)
    //val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)
    val input = sc.textFile("news.2013.en.short.lemma").map(line => line.split(" ").toSeq)
    val words = input.flatMap(x => x).map(s=>s.toLowerCase).map(Preprocessing.map2)


    //val tmp = words.mapPartitions(iter=>iter)
    // val tmp = words.collect()

    //val skipgram = new SkipGram().setNumPartitions(args(1).toInt).setNumIterations(args(2).toInt).setNegative(args(3).toInt)
    val skipgram = new SkipGram().setNumPartitions(4).setNumIterations(20).setNegative(5).setMinCount(5).setWindow(5).setVectorSize(100).setSample(0.1)
    val model = skipgram.fit(words)
    //val synonyms = model.findSynonyms(args(4), 10)

    model.save("./data_new")

    val synonyms = model.findSynonyms("apple", 40)
    //val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    println()

    //skipgram.cleanSyn()




    val msskipgram = new MSSkipGram().setSkipGram(skipgram).setNumPartitions(4).setNumIterations(40).setNegative(5).setNumSenses(2).setMinCount(5).setWindow(5).setVectorSize(100).setSample(0.1).setSentenceIter(5)
    //val msskipgram = new MSSkipGram(skipgram).setNumPartitions(args(5).toInt).setNumIterations(args(6).toInt).setNegative(args(7).toInt).setNumSenses(args(8).toInt)

    val newModel = msskipgram.fit(words)



    for (i <- 0 to 1) {
      val newSynonyms = newModel.findSynonyms("apple"+i, 40)
      //val newSynonyms = newModel.findSynonyms(args(9)+i, 10)
      //var newSynonyms = newModel.findSynonyms("day0", 10)

      println()
      for ((synonym, cosineSimilarity) <- newSynonyms) {
        println(s"$synonym $cosineSimilarity")
      }
    }

    model.save("./data_new")

    println("total time:"+(currentTime-startTime)/1000.0)
  }
}

