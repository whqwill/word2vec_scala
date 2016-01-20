package haiqing.word2vec

import java.io.{File, PrintWriter}

import com.github.fommil.netlib.BLAS._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseMatrix, BLAS, DenseVector}

/**
 * Created by hwang on 19.01.16.
 */
class Sence2Vec extends Serializable{
  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 4
  private var numIterations = 2
  private var seed = util.Random.nextLong()
  private var minCount = 5
  private var negative = 5
  private var numSenses = 2
  private var window = 5
  private var sentenceIter = 5
  private var numSentencesPerIter = 200
  def setVectorSize(vectorSize: Int): this.type = {
    this.vectorSize = vectorSize
    this
  }
  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0, s"numPartitions must be greater than 0 but got $numPartitions")
    this.numPartitions = numPartitions
    this
  }
  def setNumIterations(numIterations: Int): this.type = {
    this.numIterations = numIterations
    this
  }
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }
  def setMinCount(minCount: Int): this.type = {
    this.minCount = minCount
    this
  }
  def setNegative(negative: Int): this.type = {
    this.negative = negative
    this
  }
  def setNumSenses(numSenses: Int): this.type = {
    this.numSenses = numSenses
    this
  }
  def setWindow(window: Int): this.type = {
    this.window = window
    this
  }
  def setSentenceIter(sentenceIter: Int): this.type = {
    this.sentenceIter = sentenceIter
    this
  }
  def setNumSentencesPerIter(numSentencesPerIter: Int): this.type = {
    this.numSentencesPerIter = numSentencesPerIter
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_SENTENCE_LENGTH = 1000
  private val POWER = 0.75
  private val VARIANCE = 0.01f
  private val TABEL_SIZE = 10000
  private var vocabSize = 0
  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var trainWordsCount = 0

  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }
  private def makeTable(): Array[Int] = {
    val table = new Array[Int](TABEL_SIZE)
    var trainWordsPow = 0.0
    for (a <- 0 to vocabSize-1)
      trainWordsPow += Math.pow(vocab(a).cn, POWER)
    var i = 0
    var d1 = Math.pow(vocab(i).cn,POWER) / trainWordsPow
    for (a <- 0 to TABEL_SIZE-1) {
      table(a) = i
      if (a*1.0/TABEL_SIZE > d1) {
        i += 1
        d1 += Math.pow(vocab(i).cn, POWER) / trainWordsPow
      }
      if (i >= vocabSize)
        i = vocabSize-1
    }
    table
  }
  private def learnVocab(words: RDD[String]): Unit = {
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .map(x => VocabWord(
        x._1,
        x._2))
      .filter(_.cn >= minCount)
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }

    //for (a <- vocab.toIterator)
    //  println(a)
    println("vocabSize = " + vocabSize)
    println("trainWordsCount = " + trainWordsCount)

  }

  private def activeFunction(v0 :Array[Float], v1 :Array[Float], hyperPara: HyperPara): Double = {
    var f = blas.sdot(hyperPara.vectorSize, v0, 0, 1, v1, 0, 1)
    if (f > hyperPara.MAX_EXP)
      f = hyperPara.expTable(hyperPara.expTable.length - 1)
    else if (f < -hyperPara.MAX_EXP)
      f = hyperPara.expTable(0)
    else {
      val ind = ((f + hyperPara.MAX_EXP) * (hyperPara.expTable.size / hyperPara.MAX_EXP / 2.0)).toInt
      f = hyperPara.expTable(ind)
    }
    f
  }

  private def getScore(syn0 : Array[Array[Float]], syn1 : Array[Array[Float]], NEG: Array[Int], posW: Int, sentence: Array[Int], hyperPara: HyperPara): Double = {
    var score = 0.0
    val w = sentence(posW)
    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)

        score += math.log(activeFunction(syn0(w), syn1(u), hyperPara))

        for (d <- 0 to NEG.size-1) {
          val Z = NEG(d)
          score += math.log(1-activeFunction(syn0(Z), syn1(u), hyperPara))
        }
      }
    }
    score
  }

  private def getGradientU(syn0 : Array[Array[Float]], syn1 : Array[Array[Float]], w: Int, u: Int, NEG: Array[Int], hyperPara: HyperPara): (Array[Float], Int) = {
    val Delta = new Array[Float](hyperPara.vectorSize)
    var num = 0

    var Z = w
    var label = 1

    for (d <- -1 to NEG.size-1) {
      if (d >= 0) {
        Z = NEG(d)
        label = 0
      }

      val g = (label-activeFunction(syn0(Z), syn1(u), hyperPara)).toFloat

      blas.saxpy(hyperPara.vectorSize, g, syn0(Z), 0, 1, Delta, 0, 1)

      num += 1
    }
    (Delta, num)
  }

  private def getGradientZ(syn0 : Array[Array[Float]], syn1 : Array[Array[Float]],  posW: Int, Z: Int, label: Int, sentence: Array[Int], hyperPara: HyperPara): (Array[Float],Int) = {
    val Delta = new Array[Float](hyperPara.vectorSize)
    var num = 0
    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)
        val g = (label-activeFunction(syn0(Z), syn1(u), hyperPara)).toFloat

        blas.saxpy(hyperPara.vectorSize, g, syn1(u), 0, 1, Delta, 0, 1)
        num += 1
      }
    }
    (Delta, num)
  }

  case class HyperPara( seed: Long, EXP_TABLE_SIZE : Int, MAX_EXP : Int, TABEL_SIZE: Int, vocabSize: Int, vectorSize : Int, window: Int, negative: Int, expTable: Array[Float], table: Array[Int])

  def trainSkipGram(words: RDD[String], outputPath: String): Unit = {
    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, createExpTable(), makeTable()))

    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            if (word.nonEmpty) {
              sentence += word.get
              sentenceLength += 1
            }
          }
          sentence.result()
        }
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val numRDD = trainWordsCount/MAX_SENTENCE_LENGTH/numSentencesPerIter
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate
    val syn0Global = new Array[Array[Float]](vocabSize)
    val syn1Global = new Array[Array[Float]](vocabSize)
    for (a <- 0 to vocabSize-1) {
      syn0Global(a) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
      syn1Global(a) = new Array[Float](vectorSize)
    }

    for (k <- 1 to numRDD*numIterations) {
      println("Iteration "+k)

      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - (k-1)*1.0/numRDD/numIterations)
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + (k-1)*numSentencesPerIter*MAX_SENTENCE_LENGTH + ", alpha = " + alpha)

      println("index = "+index)
      val tmpRDD = sentenceSplit(index).mapPartitionsWithIndex { (idx,iter) =>

        //println("idx="+idx+" iter.size="+iter.size)

        util.Random.setSeed(bcHyperPara.value.seed*k+idx)
        //val newIter = mutable.MutableList[(Int,Array[Float])]()
        val synHash = new mutable.HashMap[Int,(Array[Float],Int)]

        while (iter.hasNext) {

          val sentence = iter.next()

          for (posW <- 0 to sentence.size-1) {

            val w = sentence(posW)
            val NEG = new Array[Int](bcHyperPara.value.negative)
            for (i <- 0 to bcHyperPara.value.negative - 1) {
              NEG(i) = bcHyperPara.value.table(Math.abs(util.Random.nextLong() % bcHyperPara.value.TABEL_SIZE).toInt)
              if (NEG(i) <= 0)
                NEG(i) = (Math.abs(util.Random.nextLong()) % (bcHyperPara.value.vocabSize - 1) + 1).toInt
            }

            for (posU <- posW-bcHyperPara.value.window+1 to posW+bcHyperPara.value.window-1)
              if (posU >= 0 && posU < sentence.size && posU != posW) {
                val u = sentence(posU)

                val delta = getGradientU(bcSyn0Global.value,bcSyn1Global.value,w,u,NEG,bcHyperPara.value)
                val deltaU = delta._1
                val deltaNum = delta._2

                //newIter.+=:(u+bcHyperPara.value.vocabSize,deltaU)

                val v1Option = synHash.get(u+bcHyperPara.value.vocabSize)
                var v1: Array[Float] = null
                var num = 0
                if (v1Option.isEmpty)
                  v1 = bcSyn1Global.value(u)
                else {
                  v1 = v1Option.get._1
                  num = v1Option.get._2
                }
                blas.saxpy(bcHyperPara.value.vectorSize, alpha.toFloat, deltaU, 0, 1, v1, 0, 1)
                synHash.put(u+bcHyperPara.value.vocabSize, (v1,num+deltaNum))
              }

            var Z = w
            var label = 1
            for (d <- -1 to NEG.size-1) {
              if (d >= 0) {
                Z = NEG(d)
                label = 0
              }

              val delta = getGradientZ(bcSyn0Global.value,bcSyn1Global.value,posW,Z,label,sentence,bcHyperPara.value)
              val deltaZ = delta._1
              val num = delta._2

              //newIter.+=:(Z,deltaZ)
              val v0Option = synHash.get(Z)
              var v0: Array[Float] = null
              var oldNum = 0
              if (v0Option.isEmpty)
                v0 = bcSyn0Global.value(Z)
              else {
                v0 = v0Option.get._1
                oldNum = v0Option.get._2
              }

              blas.saxpy(bcHyperPara.value.vectorSize, alpha.toFloat, deltaZ, 0, 1, v0, 0, 1)
              synHash.put(Z,(v0,oldNum+num))
            }
          }
        }
        println("idx="+idx+" synHash.size="+synHash.size)
        synHash.toIterator
        //println(newIter.size)
        //newIter.toIterator
      }.cache()

      val updateSyn = tmpRDD.reduceByKey{(v1,v2)=>for(a<-0 to v1._1.size-1)v1._1(a) += v2._1(a);(v1._1,v1._2+v2._2)}.collect()

      bcSyn0Global.unpersist()
      bcSyn1Global.unpersist()

      for (a<-0 to updateSyn.size-1) {
        val index = updateSyn(a)._1
        if (index < vocabSize)
          blas.saxpy(vectorSize, 1.0f / updateSyn(a)._2._2 , updateSyn(a)._2._1, 0, 1, syn0Global(index), 0, 1)
          //blas.saxpy(vectorSize, alpha.toFloat / countSyn.get(index).get, updateSyn(a)._2, 0, 1, syn0Global(index), 0, 1)
        else
          blas.saxpy(vectorSize, 1.0f / updateSyn(a)._2._2, updateSyn(a)._2._1, 0, 1, syn1Global(index-vocabSize), 0, 1)
          //blas.saxpy(vectorSize, alpha.toFloat / countSyn.get(index).get, updateSyn(a)._2, 0, 1, syn1Global(index-vocabSize), 0, 1)
      }

      println(syn0Global(0)(0))

    }

    val file1 = new PrintWriter(new File(outputPath+"/wordIndex.txt"))
    val file2 = new PrintWriter(new File(outputPath+"/syn0.txt"))
    val file3 = new PrintWriter(new File(outputPath+"/syn1.txt"))
    val iter = vocabHash.toIterator
    while (iter.hasNext) {
      val tmp = iter.next()
      file1.write(tmp._2+" "+tmp._1+"\n")
    }
    for (a <- 0 to vocabSize-1) {
      for (b <-0 to vectorSize-1) {
        file2.write(syn0Global(a)(b)+" ")
        file3.write(syn1Global(a)(b)+" ")
      }
      file2.write("\n")
      file3.write("\n")
    }
    file1.close()
    file2.close()
    file3.close()
  }

  def trainMSSkipGram(words: RDD[String], synPath: String, outputPath: String): Unit = {


  }
}
