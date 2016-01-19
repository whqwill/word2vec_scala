package haiqing.word2vec

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
class Sence2Vec {
  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 4
  private var numIterations = 2
  private var seed = util.Random.nextLong()
  private var minCount = 5
  private var negative = 5
  private var numSenses = 2
  private var window = 5
  private var senIter = 5
  private var numSencesPerIter = 200
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
  def setSenIter(senIter: Int): this.type = {
    this.senIter = senIter
    this
  }
  def setNumSencesPerIter(numSencesPerIter: Int): this.type = {
    this.numSencesPerIter = numSencesPerIter
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
  }

  private def activeFunction(syn0 :Array[Float], syn1 :Array[Float], hyperPara: HyperPara): Double = {
    var f = blas.sdot(hyperPara.vectorSize, syn0, 1, 1, syn1, 1, 1)
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

  private def getScore(syn0 :Array[Float], syn1 :Array[Float], NEG: Array[Int], posW: Int, sentence: Array[Int], hyperPara: HyperPara): Double = {
    var score = 0.0
    val w = sentence(posW)
    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)
        val l1 = u * hyperPara.vectorSize
        val l0 = w * hyperPara.vectorSize
        score += math.log(activeFunction(syn0, syn1, hyperPara))

        for (d <- 0 to NEG.size-1) {
          val Z = NEG(d)
          val l0 = Z * hyperPara.vectorSize
          score += math.log(1-activeFunction(syn0, syn1, hyperPara))
        }
      }
    }
    score
  }

  private def getGradientU(syn0 :Array[Float], syn1 :Array[Float], w: Int, u: Int, NEG: Array[Int], expTable: Array[Float], vectorSize: Int, MAX_EXP: Int): Array[Float] = {
    val Delta = new Array[Float](vectorSize)

    val l1 = u * vectorSize
    var Z = w
    var label = 1

    for (d <- -1 to NEG.size-1) {
      if (d >= 0) {
        Z = NEG(d)
        label = 0
      }

      val l0 = Z * vectorSize
      val g = (label-activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP)).toFloat

      blas.saxpy(vectorSize, g, syn0, l0, 1, Delta, 0, 1)
    }
    Delta
  }

  private def getGradientZ(syn0 :Array[Float], syn1 :Array[Float],  posW: Int, Z: Int, label: Int, window: Int, sentence: Array[Int], expTable: Array[Float], vectorSize: Int, MAX_EXP: Int): Array[Float] = {
    val Delta = new Array[Float](vectorSize)
    for (posU <- posW-window+1 to posW+window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)
        val l1 = u * vectorSize
        val l0 = Z * vectorSize
        val g = (label-activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP)).toFloat

        blas.saxpy(vectorSize, g, syn1, l1, 1, Delta, 0, 1)
      }
    }
    Delta
  }

  case class HyperPara( seed: Long, EXP_TABLE_SIZE : Int, MAX_EXP : Int, vocabSize: Int, vectorSize : Int, window: Int, expTable: Array[Float], table: Array[Int])

  def trainSkipGram(words: RDD[String], outputFile: String): Unit = {
    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, EXP_TABLE_SIZE, MAX_EXP, vocabSize, vectorSize, window, createExpTable(), makeTable()))

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
    val numRDD = trainWordsCount/MAX_SENTENCE_LENGTH/numSencesPerIter
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

      val tmpRDD = sentenceSplit(index).mapPartitionsWithIndex { (idx,iter) =>

        //val seed = bcHyperPara.value.seed*k+idx
        val newIter = mutable.MutableList[(Int,Array[_ >: Int with Float <: AnyVal])]()
        val syn0Hash = new mutable.HashMap[Int,Array[Float]]
        val syn1Hash = new mutable.HashMap[Int,Array[Float]]

        while (iter.hasNext) {

          val sentence = iter.next()

          for (posW <- 0 to sentence.size-1) {

            val w = sentence(posW)
            val NEG = new Array[Int](negative)
            for (i <- 0 to negative - 1) {
              NEG(i) = bcHyperPara.value.table(Math.abs(util.Random.nextLong() % TABEL_SIZE).toInt)
              if (NEG(i) <= 0)
                NEG(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt
            }

            for (posU <- posW-window+1 to posW+window-1)
              if (posU >= 0 && posU < sentence.size && posU != posW) {
                val u = sentence(posU)
                val l1 = u * vectorSize
                val syn1Option = syn1Hash.get(u)
                var syn1: Array[Float] = null
                if (syn1Option.isEmpty)
                  syn1 = bcSyn1Global.value(u)
                else
                  syn1 = syn1Option.get
                val deltaU = getGradientU(syn0,syn1,w,u,NEG,expTable.value,vectorSize,MAX_EXP)
                syn1Hash.put(u, syn1)
                blas.saxpy(vectorSize, alpha.toFloat, deltaU, 0, 1, syn1, 1, 1)
              }

            var Z = w
            var label = 1
            for (d <- -1 to NEG.size-1) {
              if (d >= 0) {
                Z = NEG(d)
                label = 0
              }

              val l0 = Z * vectorSize
              val deltaZ = getGradientZ(syn0,syn1,posW,Z,label,window,sentence,expTable.value,vectorSize,MAX_EXP)
              blas.saxpy(vectorSize, alpha.toFloat, deltaZ, 0, 1, syn0, l0, 1)
            }
          }

        }
        newIter.toIterator
      }
    }
  }
}
