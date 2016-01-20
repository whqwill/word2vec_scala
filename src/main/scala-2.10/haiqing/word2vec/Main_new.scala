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
    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq)
    val words = input.flatMap(x => x).filter(s=>s.length>0).filter(s=>s(0).isLetter).map(s => s.toLowerCase)//.map(Preprocessing.map2)

    println(words.take(100).reduce((a,b)=>a+" "+b))

    val sen2Vec = new Sence2Vec().setNumPartitions(args(1).toInt).setNumSentencesPerIter(args(2).toInt).setNumIterations(args(3).toInt).setLearningRate(args(4).toFloat).setNegative(args(5).toInt).setWindow(args(6).toInt).setMinCount(args(7).toInt).setSeed(args(8).toLong).setVectorSize(args(9).toInt)
    sen2Vec.trainSkipGram(words, args(10))
    println(words.count())
    println("total time:"+(currentTime-startTime)/1000.0)


    val model = Processing.loadModel(args(10))
    val synonyms = model.findSynonyms("day", 20)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }
    */


    val bc = sc.broadcast(new Array[Float](10))
    sc.parallelize(new Array[Float](1000)).mapPartitionsWithIndex { (idx,x)=>
      val b = bc.value.clone()
      println("idx="+idx+" before change, b(0)="+b(0))
      //if (idx == 0)
       // b(0) = 7


      if (idx == 0)
        b(0)  = 7
      println("idx="+idx+" after change, bc.value(0)="+b(0))
      x
    }.count()



  }
}
