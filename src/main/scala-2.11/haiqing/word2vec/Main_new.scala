package haiqing.word2vec

import org.apache.spark.{SparkContext, SparkConf}

import scala.compat.Platform._

/**
 * Created by hwang on 19.01.16.
 */
object Main_new {
  def main(args: Array[String]): Unit = {
    val startTime = currentTime
    val conf = new SparkConf().setAppName("Sence2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)

    println(sc.defaultParallelism + "   " + sc.master)

    /*
    val input = sc.textFile(args(0)).map(line => line.split(" ").map{x=>var end=x.size-1;while(x(end))x.substring(0,end)}.filter(x=>x.size>0)).filter(line=>line.size>1)

    val str = input.take(1)(0)

    println("size="+str.size+" sentence:")
    for(s<-str) print(s+" ")
    println()

    return
    */

    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)

    println(input.flatMap(x => x).count())

    val words = input.flatMap(x => x).map(x=>x.toLowerCase)


    //val errs = input.flatMap(x => x).filter(s=>s.length>0).filter(s=>(!s(0).isLetter))
    println(words.take(100).reduce((a,b)=>a+" "+b))


    val sen2Vec = new Sence2Vec().setNumPartitions(sc.defaultParallelism).setNumSentencesPerIterPerCore(args(1).toInt).setNumEpoch(args(2).toInt).setLearningRate(args(3).toFloat).setNegative(args(4).toInt).setWindow(args(5).toInt).setMinCount(args(6).toInt).setSeed(args(7).toLong).setVectorSize(args(8).toInt).setMAX_SENTENCE_LENGTH(args(9).toInt)

    if (args.length > 12 && args(12).toLowerCase == "local")
      sen2Vec.trainSkipGramLocal(words, args(10))
    else if (args.length > 12 && args(12).toLowerCase == "gradient")
      sen2Vec.trainSkipGramGradient(words, args(10))
    else
      sen2Vec.trainSkipGram(words, args(10))
    println(words.count())
    println("total time:"+(currentTime-startTime)/1000.0)

    val model = Processing.loadModel(args(10))
    val synonyms = model.findSynonyms(args(11), 20)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    /*
        val a = new Array[Float](10)
        for (i<-0 to 2) {
          val bc = sc.broadcast(a)
          sc.parallelize(new Array[Float](1000)).mapPartitionsWithIndex { (idx, x) =>
            bc.value(0) += 1
            x
          }.count()
          println(bc.value(0))
        }
        println(a(0))
        */
  }

}