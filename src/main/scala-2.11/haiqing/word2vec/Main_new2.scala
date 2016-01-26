package haiqing.word2vec

import org.apache.spark.{SparkConf, SparkContext}

import scala.compat.Platform._

/**
 * Created by hwang on 26.01.16.
 */
object Main_new2 {
  def main(args: Array[String]): Unit = {
    val startTime = currentTime
    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)

    println(sc.defaultParallelism + "   " + sc.master)
    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)
    val words = input.flatMap(x => x).filter(s=>s.length>0).filter(s=>s(0).isLetter).map(s => s.toLowerCase)//.map(Preprocessing.map2)

    println(words.take(100).reduce((a,b)=>a+" "+b))

    val sen2Vec = new Sence2Vec().setNumPartitions(args(1).toInt).setNumSentencesPerIter(args(2).toInt).setNumIterations(args(3).toInt).setLearningRate(args(4).toFloat).setNegative(args(5).toInt).setWindow(args(6).toInt).setMinCount(args(7).toInt).setSeed(args(8).toLong).setVectorSize(args(9).toInt).setMAX_SENTENCE_LENGTH(args(10).toInt).setNumSenses(args(11).toInt).setSentenceIter(args(12).toInt)

    sen2Vec.trainMSSkipGramLocalVariant(words, args(13))

    println(words.count())
    println("total time:"+(currentTime-startTime)/1000.0)

    val model = Processing.loadModelSenses(args(13))
    val synonyms = model.findSynonyms(args(14), 20)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

  }
}
