package haiqing.word2vec

import java.io.FileOutputStream

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

    //out = Some(new FileOutputStream("/tmp/Test.class.copy"))


    println(sc.defaultParallelism + "   " + sc.master)
    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)
    val words = input.flatMap(x => x).filter(s=>s.length>0).filter(s=>s(0).isLetter).map(s => s.toLowerCase)//.map(Preprocessing.map2)

    println(words.take(100).reduce((a,b)=>a+" "+b))

    val sen2Vec = new Sence2Vec().setNumPartitions(sc.defaultParallelism).setNumSentencesPerIterPerCore(args(1).toInt).setNumEpoch(args(2).toInt).setLearningRate(args(3).toFloat).setNegative(args(4).toInt).setWindow(args(5).toInt).setMinCount(args(6).toInt).setMultiSenseRate(args(7).toDouble).setSeed(args(8).toLong).setVectorSize(args(9).toInt).setMAX_SENTENCE_LENGTH(args(10).toInt).setNumSenses(args(11).toInt).setSentenceIter(args(12).toInt)

    sen2Vec.trainMSSkipGramLocal(words, args(13), args(13))

    println(words.count())
    println("total time:"+(currentTime-startTime)/1000.0)

    val model = Processing.loadModelSenses(args(13))
    val synonyms = model.findSynonyms(args(14), 20)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

  }
}
