package haiqing.word2vec

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable.ArrayBuilder

/**
 * Created by hwang on 18.01.16.
 */
object testSentence {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("testSentence")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
    }
    val sc = new SparkContext(conf)
    println(sc.defaultParallelism + "   " + sc.master)


    var input = sc.textFile("news.2013.en.short.lemma", 8).map(line => line.split(" ").toSeq)

    val b = input.take(4)
    println(b(0).toString())
    println(b(1))
    println(b(2))
    println(b(3))

    val words = input.flatMap(x => x).map(s=>s.toLowerCase).map(Preprocessing.map2)

    val sentences: RDD[Array[String]] = words.mapPartitions { iter =>
      println("!!!")
      new Iterator[Array[String]] {
        def hasNext: Boolean = iter.hasNext

        def next(): Array[String] = {
          val sentence = ArrayBuilder.make[String]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < 1000) {
              sentence += iter.next()
              sentenceLength += 1
          }
          sentence.result()
        }
      }
    }

    val a = sentences.take(2)
    println(a(0).toSeq)
    println(a(1).toSeq)

  }
}
