package haiqing.word2vec

import java.io.{File, PrintWriter}

import com.github.fommil.netlib.BLAS._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseMatrix, BLAS, DenseVector}

import scala.io.Source

/**
 * Created by hwang on 19.01.16.
 */
class Sence2Vec extends Serializable{
  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 4
  private var numEpoch = 2
  private var seed = util.Random.nextLong()
  private var minCount = 5
  private var negative = 5
  private var numSenses = 2
  private var window = 5
  private var sentenceIter = 5
  private var numSentencesPerIterPerCore = 200
  private var MAX_SENTENCE_LENGTH = 1000
  private var multiSenseRate = 0.5
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
  def setNumEpoch(numEpoch: Int): this.type = {
    this.numEpoch = numEpoch
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
  def setNumSentencesPerIterPerCore(numSentencesPerIterPerCore: Int): this.type = {
    this.numSentencesPerIterPerCore = numSentencesPerIterPerCore
    this
  }
  def setMAX_SENTENCE_LENGTH(MAX_SENTENCE_LENGTH: Int): this.type = {
    this.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
    this
  }
  def setMultiSenseRate(multiSenseRate: Double): this.type = {
    this.multiSenseRate = multiSenseRate
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val POWER = 0.75
  private val VARIANCE = 0.01f
  private val TABEL_SIZE = 10000
  private var vocabSize = 0
  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var trainWordsCount = 0
  private var senseTable: Array[Int] = null

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

    senseTable = new Array[Int](vocabSize)

    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }

    var numMultiSenseWords = 0
    a = 0
    while (a < vocabSize) {
      if (vocab(a).cn > trainWordsCount / vocabSize) {
        senseTable(a) = 2
        numMultiSenseWords += 1
      }
      else
        senseTable(a) = 1
      a += 1
    }

    println("numMultiSenseWords = " + numMultiSenseWords)
    println("vocabSize = " + vocabSize)
    println("trainWordsCount = " + trainWordsCount)

  }

  private def activeFunction(v0 :Array[Float], v1 :Array[Float], hyperPara: HyperPara): Double = {
    var f = blas.sdot(hyperPara.vectorSize, v0, 1, v1, 1)
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

  private def getGradientU(synHash: mutable.HashMap[Int,Array[Float]], w: Int, u: Int, NEG: Array[Int], hyperPara: HyperPara): Array[Float] = {
    val Delta = new Array[Float](hyperPara.vectorSize)

    var Z = w
    var label = 1

    for (d <- -1 to NEG.size-1) {
      if (d >= 0) {
        Z = NEG(d)
        label = 0
      }

      val g = (label-activeFunction(synHash.get(Z).get, synHash.get(u+hyperPara.vocabSize).get, hyperPara)).toFloat

      blas.saxpy(hyperPara.vectorSize, g, synHash.get(Z).get, 1, Delta, 1)

    }
    Delta
  }

  private def getGradientZ(synHash: mutable.HashMap[Int,Array[Float]],  posW: Int, Z: Int, label: Int, sentence: Array[Int], hyperPara: HyperPara): Array[Float] = {
    val Delta = new Array[Float](hyperPara.vectorSize)

    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)
        val g = (label-activeFunction(synHash.get(Z).get, synHash.get(u+hyperPara.vocabSize).get, hyperPara)).toFloat

        blas.saxpy(hyperPara.vectorSize, g, synHash.get(u+hyperPara.vocabSize).get, 1, Delta, 1)
      }
    }
    Delta
  }

  case class HyperPara( seed: Long, MAX_SENTENCE_LENGTH: Int, EXP_TABLE_SIZE : Int, MAX_EXP : Int, TABEL_SIZE: Int, vocabSize: Int, vectorSize : Int, window: Int, negative: Int, numSenses: Int, sentenceIter: Int, expTable: Array[Float], table: Array[Int])

  def trainSkipGram(words: RDD[String], outputPath: String): Unit = {
    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, MAX_SENTENCE_LENGTH, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, numSenses, sentenceIter, createExpTable(), makeTable()))

    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < bcHyperPara.value.MAX_SENTENCE_LENGTH) {
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
    val numRDD = trainWordsCount/MAX_SENTENCE_LENGTH/(numSentencesPerIterPerCore*numPartitions)
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate
    val syn0Global = new Array[Array[Float]](vocabSize)
    val syn1Global = new Array[Array[Float]](vocabSize)
    for (a <- 0 to vocabSize-1) {
      syn0Global(a) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
      syn1Global(a) = new Array[Float](vectorSize)
    }


    println("numRDD="+numRDD)
    println()

    var wordCount = 0

    for (k <- 1 to numRDD*numEpoch) {
      println("Iteration "+k)

      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - wordCount/(trainWordsCount*numEpoch))
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + wordCount + ", alpha = " + alpha)

      val accWordCount = sc.accumulator(wordCount)

      println("index = "+index)
      val tmpRDD = sentenceSplit(index).mapPartitionsWithIndex { (idx,iter) =>

        //println("idx="+idx+" iter.size="+iter.size)

        util.Random.setSeed(bcHyperPara.value.seed*k+idx)
        //val newIter = mutable.MutableList[(Int,Array[Float])]()
        val synHash = new mutable.HashMap[Int,Array[Float]]

        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

          for (posW <- 0 to sentence.size-1) {

            val w = sentence(posW)
            if (synHash.get(w).isEmpty)
              synHash.put(w, syn0(w).clone())

            val NEG = new Array[Int](bcHyperPara.value.negative)
            for (i <- 0 to bcHyperPara.value.negative - 1) {
              NEG(i) = bcHyperPara.value.table(Math.abs(util.Random.nextLong() % bcHyperPara.value.TABEL_SIZE).toInt)
              if (NEG(i) <= 0)
                NEG(i) = (Math.abs(util.Random.nextLong()) % (bcHyperPara.value.vocabSize - 1) + 1).toInt

              val Z = NEG(i)
              if (synHash.get(Z).isEmpty)
                synHash.put(Z, syn0(Z).clone())
            }

            for (posU <- posW-bcHyperPara.value.window+1 to posW+bcHyperPara.value.window-1)
              if (posU >= 0 && posU < sentence.size && posU != posW) {
                val u = sentence(posU)

                if (synHash.get(u+hyperPara.vocabSize).isEmpty)
                  synHash.put(u+hyperPara.vocabSize, syn1(u).clone())

                val deltaU = getGradientU(synHash,w,u,NEG,hyperPara)

                val v1 = synHash.get(u+hyperPara.vocabSize).get
                blas.saxpy(bcHyperPara.value.vectorSize, alpha.toFloat, deltaU, 1, v1, 1)
              }

            var Z = w
            var label = 1
            for (d <- -1 to NEG.size-1) {
              if (d >= 0) {
                Z = NEG(d)
                label = 0
              }

              val deltaZ = getGradientZ(synHash,posW,Z,label,sentence,hyperPara)

              val v0 = synHash.get(Z).get
              blas.saxpy(bcHyperPara.value.vectorSize, alpha.toFloat, deltaZ, 1, v0, 1)

            }
          }
        }
        println("idx="+idx+" synHash.size="+synHash.size)
        synHash.toIterator
        //println(newIter.size)
        //newIter.toIterator
      }.cache()

      val updateSyn = tmpRDD.reduceByKey{(v1,v2)=>for(a<-0 to v1.size-1)v1(a) += v2(a);v1}.collect()
      val countSyn = tmpRDD.countByKey()

      bcSyn0Global.unpersist()
      bcSyn1Global.unpersist()

      for (a<-0 to updateSyn.size-1) {
        val index = updateSyn(a)._1
        if (index == 10)
          println("index=10 "+countSyn(index))
        if (index == 100)
          println("index=100 "+countSyn(index))
        if (index < vocabSize) {
          for (i <- 0 to vectorSize-1)
            syn0Global(index)(i) = 0.0f

          blas.saxpy(vectorSize, 1.0f / countSyn(index), updateSyn(a)._2, 1, syn0Global(index), 1)
          //blas.saxpy(vectorSize, alpha.toFloat / countSyn.get(index).get, updateSyn(a)._2, 0, 1, syn0Global(index), 0, 1)
        }
        else {
          for (i <- 0 to vectorSize-1)
            syn1Global(index - vocabSize)(i) = 0.0f

          blas.saxpy(vectorSize, 1.0f / countSyn(index), updateSyn(a)._2, 1, syn1Global(index - vocabSize), 1)
          //blas.saxpy(vectorSize, alpha.toFloat / countSyn.get(index).get, updateSyn(a)._2, 0, 1, syn1Global(index-vocabSize), 0, 1)
        }
      }

      println(syn0Global(0)(0))

      wordCount = accWordCount.value
    }

    writeToFile(outputPath, syn0Global, syn1Global)
  }

  def trainSkipGramGradient(words: RDD[String], outputPath: String): Unit = {
    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, MAX_SENTENCE_LENGTH, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, numSenses, sentenceIter, createExpTable(), makeTable()))

    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < bcHyperPara.value.MAX_SENTENCE_LENGTH) {
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
    val numRDD = trainWordsCount/MAX_SENTENCE_LENGTH/(numSentencesPerIterPerCore*numPartitions)
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate
    val syn0Global = new Array[Array[Float]](vocabSize)
    val syn1Global = new Array[Array[Float]](vocabSize)
    for (a <- 0 to vocabSize-1) {
      syn0Global(a) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
      syn1Global(a) = new Array[Float](vectorSize)
    }


    println("numRDD="+numRDD)
    println()

    var wordCount = 0

    for (k <- 1 to numRDD*numEpoch) {
      println("Iteration "+k)

      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - wordCount/(trainWordsCount*numEpoch))
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + wordCount + ", alpha = " + alpha)

      val accWordCount = sc.accumulator(wordCount)

      println("index = "+index)
      val tmpRDD = sentenceSplit(index).mapPartitionsWithIndex { (idx,iter) =>

        //println("idx="+idx+" iter.size="+iter.size)

        util.Random.setSeed(bcHyperPara.value.seed*k+idx)
        //val newIter = mutable.MutableList[(Int,Array[Float])]()
        val synHash = new mutable.HashMap[Int,(Array[Float],Int)]

        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

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



                val Pair = getGradientULocalPair(syn0,syn1,w,u,NEG,hyperPara)
                val deltaU = Pair._1
                val addNum = Pair._2


                val v1Option = synHash.get(u+hyperPara.vocabSize)
                var v1 : Array[Float] = null
                var num = 0
                if (v1Option.isEmpty)
                  v1 = new Array[Float](hyperPara.vectorSize)
                else {
                  v1 = v1Option.get._1
                  num = v1Option.get._2
                }
                blas.saxpy(hyperPara.vectorSize, 1.0f, deltaU, 1, v1, 1)
                synHash.put(u+hyperPara.vocabSize, (v1,num+1))
              }

            var Z = w
            var label = 1
            for (d <- -1 to NEG.size-1) {
              if (d >= 0) {
                Z = NEG(d)
                label = 0
              }

              val Pair = getGradientZLocalPair(syn0,syn1,posW,Z,label,sentence,hyperPara)
              val deltaZ = Pair._1
              val addNum = Pair._2

              val v0Option = synHash.get(Z)
              var v0 : Array[Float] = null
              var num = 0
              if (v0Option.isEmpty)
                v0 = new Array[Float](hyperPara.vectorSize)
              else {
                v0 = v0Option.get._1
                num = v0Option.get._2
              }
              blas.saxpy(hyperPara.vectorSize, 1.0f, deltaZ, 1, v0, 1)
              synHash.put(Z, (v0,num+1))
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

      var countGradient = 0

      for (a<-0 to updateSyn.size-1) {
        countGradient += updateSyn(a)._2._2
        val index = updateSyn(a)._1
        if (index == 10)
          println("index=10 count="+updateSyn(a)._2._2)
        if (index == vocabSize+100)
          println("index=vocabSize+100 count="+updateSyn(a)._2._2)
        if (index < vocabSize)
          blas.saxpy(vectorSize, alpha.toFloat/updateSyn(a)._2._2, updateSyn(a)._2._1, 1, syn0Global(index), 1)
        else
          blas.saxpy(vectorSize, alpha.toFloat/updateSyn(a)._2._2, updateSyn(a)._2._1, 1, syn1Global(index - vocabSize), 1)
      }

      println(syn0Global(0)(0))
      println("words per core: "+((accWordCount.value-wordCount).toFloat/numPartitions).toInt)
      println("gradients per core: "+(countGradient.toFloat/numPartitions).toInt)
      wordCount = accWordCount.value

    }

    writeToFile(outputPath, syn0Global, syn1Global)
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
          //words should be different
          if (Z % hyperPara.vocabSize != w % hyperPara.vocabSize)
            score += math.log(1-activeFunction(syn0(Z), syn1(u), hyperPara))
        }
      }
    }
    score
  }

  private def getGradientULocal(syn0: Array[Array[Float]], syn1: Array[Array[Float]], w: Int, u: Int, NEG: Array[Int], hyperPara: HyperPara): Array[Float] = {
    val Delta = new Array[Float](hyperPara.vectorSize)

    var Z = w
    var label = 1

    for (d <- -1 to NEG.size-1) {
      if (d >= 0) {
        Z = NEG(d)
        label = 0
      }

      val g = (label-activeFunction(syn0(Z), syn1(u), hyperPara)).toFloat

      blas.saxpy(hyperPara.vectorSize, g, syn0(Z), 1, Delta, 1)

    }
    Delta
  }

  private def getGradientZLocal(syn0: Array[Array[Float]], syn1: Array[Array[Float]],  posW: Int, Z: Int, label: Int, sentence: Array[Int], hyperPara: HyperPara): Array[Float] = {
    val Delta = new Array[Float](hyperPara.vectorSize)

    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)
        val g = (label-activeFunction(syn0(Z), syn1(u), hyperPara)).toFloat

        blas.saxpy(hyperPara.vectorSize, g, syn1(u), 1, Delta, 1)
      }
    }
    Delta
  }

  private def getGradientULocalPair(syn0: Array[Array[Float]], syn1: Array[Array[Float]], w: Int, u: Int, NEG: Array[Int], hyperPara: HyperPara): (Array[Float],Int) = {
    val Delta = new Array[Float](hyperPara.vectorSize)

    var Z = w
    var label = 1
    var num=0
    for (d <- -1 to NEG.size-1) {
      if (d >= 0) {
        Z = NEG(d)
        label = 0
      }

      val g = (label-activeFunction(syn0(Z), syn1(u), hyperPara)).toFloat

      blas.saxpy(hyperPara.vectorSize, g, syn0(Z), 1, Delta, 1)
      num+=1
    }
    (Delta,num)
  }

  private def getGradientZLocalPair(syn0: Array[Array[Float]], syn1: Array[Array[Float]],  posW: Int, Z: Int, label: Int, sentence: Array[Int], hyperPara: HyperPara): (Array[Float],Int) = {
    val Delta = new Array[Float](hyperPara.vectorSize)
    var num=0
    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)
        val g = (label-activeFunction(syn0(Z), syn1(u), hyperPara)).toFloat

        blas.saxpy(hyperPara.vectorSize, g, syn1(u), 1, Delta, 1)
        num+=1
      }
    }
    (Delta,num)
  }

  def trainSkipGramLocal(words: RDD[String], outputPath: String): Unit = {
    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, MAX_SENTENCE_LENGTH, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, numSenses, sentenceIter, createExpTable(), makeTable()))

    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < bcHyperPara.value.MAX_SENTENCE_LENGTH) {
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
    val numRDD = trainWordsCount/MAX_SENTENCE_LENGTH/(numSentencesPerIterPerCore*numPartitions)
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate
    val syn0Global = new Array[Array[Float]](vocabSize)
    val syn1Global = new Array[Array[Float]](vocabSize)
    for (a <- 0 to vocabSize-1) {
      syn0Global(a) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
      syn1Global(a) = new Array[Float](vectorSize)
    }


    println("numRDD="+numRDD)
    println()

    var wordCount = 0
    val bcSyn0Global = sc.broadcast(syn0Global)
    val bcSyn1Global = sc.broadcast(syn1Global)

    for (k <- 1 to numRDD*numEpoch) {
      println("Iteration "+k)


      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - wordCount*1.0f/(trainWordsCount*numEpoch))
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + wordCount + ", alpha = " + alpha)

      val accWordCount = sc.accumulator(wordCount)

      println("index = "+index)
      //newSentences.sample(false, 1.0/numRDD).foreachPartition { iter =>
      sentenceSplit(index).foreachPartition { iter =>

        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

          for (posW <- 0 to sentence.size-1) {

            val w = sentence(posW)

            val NEG = new Array[Int](hyperPara.negative)
            for (i <- 0 to hyperPara.negative - 1) {
              NEG(i) = hyperPara.table(Math.abs(util.Random.nextLong() % hyperPara.TABEL_SIZE).toInt)
              if (NEG(i) <= 0)
                NEG(i) = (Math.abs(util.Random.nextLong()) % (hyperPara.vocabSize - 1) + 1).toInt
            }

            for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1)
              if (posU >= 0 && posU < sentence.size && posU != posW) {
                val u = sentence(posU)

                val deltaU = getGradientULocal(syn0,syn1,w,u,NEG,hyperPara)

                blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaU, 1, syn1(u), 1)
              }

            var Z = w
            var label = 1
            for (d <- -1 to NEG.size-1) {
              if (d >= 0) {
                Z = NEG(d)
                label = 0
              }

              val deltaZ = getGradientZLocal(syn0,syn1,posW,Z,label,sentence,hyperPara)

              blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaZ, 1, syn0(Z), 1)

            }
          }
        }
      }

      println(syn0Global(0)(0))

      wordCount = accWordCount.value

    }

    writeToFile(outputPath, syn0Global, syn1Global)
  }

  def trainMSSkipGramLocal(words: RDD[String], synPath:String, outputPath: String): Unit = {

    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, MAX_SENTENCE_LENGTH, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, numSenses, sentenceIter, createExpTable(), makeTable()))

    vocab.toSeq.sortBy(_.cn).toArray

    //generate sentences from words (senses are generated randomly)
    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < bcHyperPara.value.MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            if (word.nonEmpty) {
              sentence += word.get + bcHyperPara.value.vocabSize * util.Random.nextInt(bcHyperPara.value.numSenses)
              sentenceLength += 1
            }
          }
          sentence.result()
        }
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val numRDD = trainWordsCount/MAX_SENTENCE_LENGTH/(numSentencesPerIterPerCore*numPartitions)
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate

    val syn0Single = Source.fromFile(synPath+"/syn0.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s=>s.toFloat).toArray
    val syn1Single = Source.fromFile(synPath+"/syn1.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s=>s.toFloat).toArray
    val syn0Global = new Array[Array[Array[Float]]](vocabSize)
    val syn1Global = new Array[Array[Array[Float]]](vocabSize)
    for (w <- 0 to vocabSize-1) {

      syn0Global(w) = new Array[Array[Float]](senseTable(w))

      for (s <- 0 to senseTable(w) - 1) {

        for (i <- 0 to vectorSize-1) {
          syn0Global(w)(s)(i) = syn0Single(w*vectorSize+i) + (util.Random.nextFloat() - 0.5f) * VARIANCE
          syn1Global(w)(s)(i) = syn1Single(w*vectorSize+i) + (util.Random.nextFloat() - 0.5f) * VARIANCE
        }
      }
    }

    val bcSyn0Global = sc.broadcast(syn0Global)
    val bcSyn1Global = sc.broadcast(syn1Global)

    println("numRDD="+numRDD)
    println()

    var wordCount = 0

    for (k <- 1 to numRDD*numEpoch) {
      println("Iteration "+k)

      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - wordCount/(trainWordsCount*numEpoch))
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + wordCount + ", alpha = " + alpha)

      val accWordCount = sc.accumulator(wordCount)

      println("index = "+index)


      //assign senses
      val tmpRDD = sentenceSplit(index).mapPartitions { iter =>

        var sumIter = 0
        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        val newIter = mutable.MutableList[Array[Int]]()

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

          var flag = false

          var sIter = 1

          while (sIter <= hyperPara.sentenceIter && (sIter == 1 || flag)) {

            for (posW <- 0 to sentence.size - 1) {

              var w = sentence(posW)

              //generate negative sampling (senses are generated randomly)
              val NEG = new Array[Int](hyperPara.negative)
              for (i <- 0 to hyperPara.negative - 1) {
                NEG(i) = hyperPara.table(Math.abs(util.Random.nextLong() % hyperPara.TABEL_SIZE).toInt)
                if (NEG(i) <= 0)
                  NEG(i) = (Math.abs(util.Random.nextLong()) % (hyperPara.vocabSize - 1) + 1).toInt
                NEG(i) = NEG(i) + hyperPara.vocabSize*util.Random.nextInt(hyperPara.numSenses)
              }


              /*
              //select the best sense
              sentence(posW) = sentence(posW)%vocabSize
              var bestSense = 0
              var bestScore = getScore(syn0,syn1,NEG,posW,sentence,hyperPara)
              for (sense <- 1 to numSenses - 1) {
                sentence(posW) = sentence(posW)%vocabSize + vocabSize*sense
                val tmpScore = getScore(syn0,syn1,NEG,posW,sentence,hyperPara)
                if (tmpScore > bestScore) {
                  bestScore = tmpScore
                  bestSense = sense
                }
              }
              sentence(posW) = sentence(posW)%vocabSize + vocabSize*bestSense

              //record if center word's sense is changed
              if (sentence(posW) != w)
                flag = true
              w = sentence(posW)

              //update syn1 (syn1 is for predicting surrounding word u)
              for (posU <- posW - hyperPara.window + 1 to posW + hyperPara.window - 1)
                if (posU >= 0 && posU < sentence.size && posU != posW) {
                  val u = sentence(posU)
                  val deltaU = getGradientULocal(syn0, syn1, w, u, NEG, hyperPara)
                  blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaU, 1, syn1(u), 1)
                }

              //update syn0 (syn0 is for vectors of center word Z)
              var Z = w
              var label = 1
              for (d <- -1 to NEG.size - 1) {
                if (d >= 0) {
                  Z = NEG(d)
                  label = 0
                }
                val deltaZ = getGradientZLocal(syn0, syn1, posW, Z, label, sentence, hyperPara)
                blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaZ, 1, syn0(Z), 1)
              }
*/
            }
            sIter+=1
          }
          sumIter += sIter
          newIter+=sentence
        }
        println("newIter.size="+newIter.size+" sumSentenceIter="+sumIter)
        newIter.toIterator
      }.cache()
      println("tmpRDD.count()="+tmpRDD.count())
      sentenceSplit(index).unpersist()
      sentenceSplit(index) = tmpRDD

      //learn parameters
      sentenceSplit(index).foreachPartition { iter =>

        var sumIter = 0
        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        val newIter = mutable.MutableList[Array[Int]]()

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

          var flag = false

          var sIter = 1

          while (sIter <= hyperPara.sentenceIter && (sIter == 1 || flag)) {

            for (posW <- 0 to sentence.size - 1) {

              var w = sentence(posW)

              //generate negative sampling (senses are generated randomly)
              val NEG = new Array[Int](hyperPara.negative)
              for (i <- 0 to hyperPara.negative - 1) {
                NEG(i) = hyperPara.table(Math.abs(util.Random.nextLong() % hyperPara.TABEL_SIZE).toInt)
                if (NEG(i) <= 0)
                  NEG(i) = (Math.abs(util.Random.nextLong()) % (hyperPara.vocabSize - 1) + 1).toInt
                NEG(i) = NEG(i) + hyperPara.vocabSize*util.Random.nextInt(hyperPara.numSenses)
              }


              /*
              //select the best sense
              sentence(posW) = sentence(posW)%vocabSize
              var bestSense = 0
              var bestScore = getScore(syn0,syn1,NEG,posW,sentence,hyperPara)
              for (sense <- 1 to numSenses - 1) {
                sentence(posW) = sentence(posW)%vocabSize + vocabSize*sense
                val tmpScore = getScore(syn0,syn1,NEG,posW,sentence,hyperPara)
                if (tmpScore > bestScore) {
                  bestScore = tmpScore
                  bestSense = sense
                }
              }
              sentence(posW) = sentence(posW)%vocabSize + vocabSize*bestSense

              //record if center word's sense is changed
              if (sentence(posW) != w)
                flag = true
              w = sentence(posW)

              //update syn1 (syn1 is for predicting surrounding word u)
              for (posU <- posW - hyperPara.window + 1 to posW + hyperPara.window - 1)
                if (posU >= 0 && posU < sentence.size && posU != posW) {
                  val u = sentence(posU)
                  val deltaU = getGradientULocal(syn0, syn1, w, u, NEG, hyperPara)
                  blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaU, 1, syn1(u), 1)
                }

              //update syn0 (syn0 is for vectors of center word Z)
              var Z = w
              var label = 1
              for (d <- -1 to NEG.size - 1) {
                if (d >= 0) {
                  Z = NEG(d)
                  label = 0
                }
                val deltaZ = getGradientZLocal(syn0, syn1, posW, Z, label, sentence, hyperPara)
                blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaZ, 1, syn0(Z), 1)
              }
*/
            }
            sIter+=1
          }
          sumIter += sIter
          newIter+=sentence
        }
        println("newIter.size="+newIter.size+" sumSentenceIter="+sumIter)
        newIter.toIterator
      }

      println("syn0Global(0)(0)="+syn0Global(0)(0))

      println("words per core: "+((accWordCount.value-wordCount).toFloat/numPartitions).toInt)
      wordCount = accWordCount.value
    }

    //writeToFileWithSenses(outputPath, syn0Global, syn1Global)
  }
  def trainMSSkipGramLocal(words: RDD[String], outputPath: String): Unit = {

    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, MAX_SENTENCE_LENGTH, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, numSenses, sentenceIter, createExpTable(), makeTable()))

    //generate sentences from words (senses are generated randomly)
    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < bcHyperPara.value.MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            if (word.nonEmpty) {
              sentence += word.get + bcHyperPara.value.vocabSize * util.Random.nextInt(bcHyperPara.value.numSenses)
              sentenceLength += 1
            }
          }
          sentence.result()
        }
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val numRDD = (trainWordsCount/MAX_SENTENCE_LENGTH)/(numSentencesPerIterPerCore*numPartitions)
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate

    val syn0Global = new Array[Array[Float]](vocabSize*numSenses)
    val syn1Global = new Array[Array[Float]](vocabSize*numSenses)
    for (a <- 0 to vocabSize*numSenses-1) {
      syn0Global(a) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
      syn1Global(a) = new Array[Float](vectorSize)
    }

    val bcSyn0Global = sc.broadcast(syn0Global)
    val bcSyn1Global = sc.broadcast(syn1Global)

    println("numRDD="+numRDD)
    println()

    var wordCount = 0

    for (k <- 1 to numRDD*numEpoch) {
      println("Iteration "+k)

      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - wordCount/(trainWordsCount*numEpoch))
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + wordCount + ", alpha = " + alpha)

      val accWordCount = sc.accumulator(wordCount)

      println("index = "+index)

      //newSentences.sample(false, 1.0/numRDD).foreachPartition { iter =>
      val tmpRDD = sentenceSplit(index).mapPartitions { iter =>

        var sumIter = 0
        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        val newIter = mutable.MutableList[Array[Int]]()

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

          var flag = false

          var sIter = 1

          while (sIter <= hyperPara.sentenceIter && (sIter == 1 || flag)) {

            for (posW <- 0 to sentence.size - 1) {

              var w = sentence(posW)

              //generate negative sampling (senses are generated randomly)
              val NEG = new Array[Int](hyperPara.negative)
              for (i <- 0 to hyperPara.negative - 1) {
                NEG(i) = hyperPara.table(Math.abs(util.Random.nextLong() % hyperPara.TABEL_SIZE).toInt)
                if (NEG(i) <= 0)
                  NEG(i) = (Math.abs(util.Random.nextLong()) % (hyperPara.vocabSize - 1) + 1).toInt
                NEG(i) = NEG(i) + hyperPara.vocabSize*util.Random.nextInt(hyperPara.numSenses)
              }

              //select the best sense
              sentence(posW) = sentence(posW)%vocabSize
              var bestSense = 0
              var bestScore = getScore(syn0,syn1,NEG,posW,sentence,hyperPara)
              for (sense <- 1 to numSenses - 1) {
                sentence(posW) = sentence(posW)%vocabSize + vocabSize*sense
                val tmpScore = getScore(syn0,syn1,NEG,posW,sentence,hyperPara)
                if (tmpScore > bestScore) {
                  bestScore = tmpScore
                  bestSense = sense
                }
              }
              sentence(posW) = sentence(posW)%vocabSize + vocabSize*bestSense

              //record if center word's sense is changed
              if (sentence(posW) != w)
                flag = true
              w = sentence(posW)

              //update syn1 (syn1 is for predicting surrounding word u)
              for (posU <- posW - hyperPara.window + 1 to posW + hyperPara.window - 1)
                if (posU >= 0 && posU < sentence.size && posU != posW) {
                  val u = sentence(posU)
                  val deltaU = getGradientULocal(syn0, syn1, w, u, NEG, hyperPara)
                  blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaU, 1, syn1(u), 1)
                }

              //update syn0 (syn0 is for vectors of center word Z)
              var Z = w
              var label = 1
              for (d <- -1 to NEG.size - 1) {
                if (d >= 0) {
                  Z = NEG(d)
                  label = 0
                }
                val deltaZ = getGradientZLocal(syn0, syn1, posW, Z, label, sentence, hyperPara)
                blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaZ, 1, syn0(Z), 1)
              }

            }
            sIter+=1
          }
          sumIter += sIter
          newIter+=sentence
        }
        println("newIter.size="+newIter.size+" sumSentenceIter="+sumIter)
        newIter.toIterator
      }.cache()

      println("tmpRDD.count()="+tmpRDD.count())

      sentenceSplit(index).unpersist()
      sentenceSplit(index) = tmpRDD

      println("syn0Global(0)(0)="+syn0Global(0)(0))

      wordCount = accWordCount.value
    }

    writeToFileWithSenses(outputPath, syn0Global, syn1Global)
  }

  private def getScoreVariant(syn0 : Array[Array[Float]], syn1 : Array[Array[Float]], NEG: Array[Int], posW: Int, sentence: Array[Int], hyperPara: HyperPara): Double = {
    var score = 0.0
    val w = sentence(posW)
    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)

        score += math.log(activeFunction(syn0(w), syn1(u%hyperPara.vocabSize), hyperPara))

        for (d <- 0 to NEG.size-1) {
          val Z = NEG(d)
          //words should be different
          if (Z % hyperPara.vocabSize != w % hyperPara.vocabSize)
            score += math.log(1-activeFunction(syn0(Z), syn1(u%hyperPara.vocabSize), hyperPara))
        }
      }
    }
    score
  }

  private def getGradientULocalVariant(syn0: Array[Array[Float]], syn1: Array[Array[Float]], w: Int, u: Int, NEG: Array[Int], hyperPara: HyperPara): Array[Float] = {
    val Delta = new Array[Float](hyperPara.vectorSize)

    var Z = w
    var label = 1

    for (d <- -1 to NEG.size-1) {
      if (d >= 0) {
        Z = NEG(d)
        label = 0
      }

      val g = (label-activeFunction(syn0(Z), syn1(u%hyperPara.vocabSize), hyperPara)).toFloat

      blas.saxpy(hyperPara.vectorSize, g, syn0(Z), 1, Delta, 1)

    }
    Delta
  }

  private def getGradientZLocalVariant(syn0: Array[Array[Float]], syn1: Array[Array[Float]],  posW: Int, Z: Int, label: Int, sentence: Array[Int], hyperPara: HyperPara): Array[Float] = {
    val Delta = new Array[Float](hyperPara.vectorSize)

    for (posU <- posW-hyperPara.window+1 to posW+hyperPara.window-1) {
      if (posU >= 0 && posU < sentence.size && posU != posW) {
        val u = sentence(posU)
        val g = (label-activeFunction(syn0(Z), syn1(u%hyperPara.vocabSize), hyperPara)).toFloat

        blas.saxpy(hyperPara.vectorSize, g, syn1(u%hyperPara.vocabSize), 1, Delta, 1)
      }
    }
    Delta
  }

  def trainMSSkipGramLocalVariant(words: RDD[String], synPath:String, outputPath: String): Unit = {

    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, MAX_SENTENCE_LENGTH, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, numSenses, sentenceIter, createExpTable(), makeTable()))

    //generate sentences from words (senses are generated randomly)
    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < bcHyperPara.value.MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            if (word.nonEmpty) {
              sentence += word.get + bcHyperPara.value.vocabSize * util.Random.nextInt(bcHyperPara.value.numSenses)
              sentenceLength += 1
            }
          }
          sentence.result()
        }
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val numRDD = trainWordsCount/MAX_SENTENCE_LENGTH/numSentencesPerIterPerCore
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate

    val syn0Single = Source.fromFile(synPath+"/syn0.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s=>s.toFloat).toArray
    val syn1Single = Source.fromFile(synPath+"/syn1.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s=>s.toFloat).toArray
    val syn0Global = new Array[Array[Float]](vocabSize*numSenses)
    val syn1Global = new Array[Array[Float]](vocabSize)
    for (w <- 0 to vocabSize-1) {

      syn1Global(w) = new Array[Float](vectorSize)
      for (i <- 0 to vectorSize-1)
        syn1Global(w)(i) = syn1Single(w*vectorSize+i)

      for (s <- 0 to numSenses - 1) {
        syn0Global(w+vocabSize*s) = new Array[Float](vectorSize)
        for (i <- 0 to vectorSize-1)
          syn0Global(w+vocabSize*s)(i) = syn0Single(w*vectorSize+i) + (util.Random.nextFloat() - 0.5f) * VARIANCE
      }


    }

    val bcSyn0Global = sc.broadcast(syn0Global)
    val bcSyn1Global = sc.broadcast(syn1Global)

    println("numRDD="+numRDD)
    println()

    var wordCount = 0

    for (k <- 1 to numRDD*numEpoch) {
      println("Iteration "+k)


      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - wordCount/(trainWordsCount*numEpoch))
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + wordCount + ", alpha = " + alpha)

      val accWordCount = sc.accumulator(wordCount)

      println("index = "+index)

      //newSentences.sample(false, 1.0/numRDD).foreachPartition { iter =>
      val tmpRDD = sentenceSplit(index).mapPartitions { iter =>

        var sumIter = 0
        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        val newIter = mutable.MutableList[Array[Int]]()

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

          var flag = false

          var sIter = 1

          while (sIter <= hyperPara.sentenceIter && (sIter == 1 || flag)) {

            for (posW <- 0 to sentence.size - 1) {

              var w = sentence(posW)

              //generate negative sampling (senses are generated randomly)
              val NEG = new Array[Int](hyperPara.negative)
              for (i <- 0 to hyperPara.negative - 1) {
                NEG(i) = hyperPara.table(Math.abs(util.Random.nextLong() % hyperPara.TABEL_SIZE).toInt)
                if (NEG(i) <= 0)
                  NEG(i) = (Math.abs(util.Random.nextLong()) % (hyperPara.vocabSize - 1) + 1).toInt
                NEG(i) = NEG(i) + hyperPara.vocabSize*util.Random.nextInt(hyperPara.numSenses)
              }

              //select the best sense
              sentence(posW) = sentence(posW)%vocabSize
              var bestSense = 0
              var bestScore = getScoreVariant(syn0,syn1,NEG,posW,sentence,hyperPara)
              for (sense <- 1 to numSenses - 1) {
                sentence(posW) = sentence(posW)%vocabSize + vocabSize*sense
                val tmpScore = getScoreVariant(syn0,syn1,NEG,posW,sentence,hyperPara)
                if (tmpScore > bestScore) {
                  bestScore = tmpScore
                  bestSense = sense
                }
              }
              sentence(posW) = sentence(posW)%vocabSize + vocabSize*bestSense

              //record if center word's sense is changed
              if (sentence(posW) != w)
                flag = true
              w = sentence(posW)

              //update syn1 (syn1 is for predicting surrounding word u)
              for (posU <- posW - hyperPara.window + 1 to posW + hyperPara.window - 1)
                if (posU >= 0 && posU < sentence.size && posU != posW) {
                  val u = sentence(posU)
                  val deltaU = getGradientULocalVariant(syn0, syn1, w, u, NEG, hyperPara)
                  blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaU, 1, syn1(u%hyperPara.vocabSize), 1)
                }

              //update syn0 (syn0 is for vectors of center word Z)
              var Z = w
              var label = 1
              for (d <- -1 to NEG.size - 1) {
                if (d >= 0) {
                  Z = NEG(d)
                  label = 0
                }
                val deltaZ = getGradientZLocalVariant(syn0, syn1, posW, Z, label, sentence, hyperPara)
                blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaZ, 1, syn0(Z), 1)
              }

            }
            sIter+=1
          }
          sumIter += sIter
          newIter+=sentence
        }
        println("newIter.size="+newIter.size+" sumSentenceIter="+sumIter)
        newIter.toIterator
      }.cache()

      println("tmpRDD.count()="+tmpRDD.count())

      sentenceSplit(index).unpersist()
      sentenceSplit(index) = tmpRDD

      println("syn0Global(0)(0)="+syn0Global(0)(0))

      wordCount = accWordCount.value
    }

    writeToFileWithSensesVariant(outputPath, syn0Global, syn1Global)
  }

  def trainMSSkipGramLocalVariant(words: RDD[String], outputPath: String): Unit = {

    val sc = words.context
    learnVocab(words)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcHyperPara = sc.broadcast(HyperPara(seed, MAX_SENTENCE_LENGTH, EXP_TABLE_SIZE, MAX_EXP, TABEL_SIZE, vocabSize, vectorSize, window, negative, numSenses, sentenceIter, createExpTable(), makeTable()))

    //generate sentences from words (senses are generated randomly)
    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < bcHyperPara.value.MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            if (word.nonEmpty) {
              sentence += word.get + bcHyperPara.value.vocabSize * util.Random.nextInt(bcHyperPara.value.numSenses)
              sentenceLength += 1
            }
          }
          sentence.result()
        }
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val numRDD = (trainWordsCount/MAX_SENTENCE_LENGTH)/(numSentencesPerIterPerCore*numPartitions)
    val sentenceSplit = newSentences.randomSplit(new Array[Double](numRDD).map(x=>x+1))
    var alpha = learningRate

    val syn0Global = new Array[Array[Float]](vocabSize*numSenses)
    val syn1Global = new Array[Array[Float]](vocabSize)

    for (a <- 0 to vocabSize*numSenses-1) {
      syn0Global(a) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
    }

    for (a <- 0 to vocabSize-1) {
      syn1Global(a) = new Array[Float](vectorSize)
    }

    val bcSyn0Global = sc.broadcast(syn0Global)
    val bcSyn1Global = sc.broadcast(syn1Global)

    println("numRDD="+numRDD)
    println()

    var wordCount = 0

    for (k <- 1 to numRDD*numEpoch) {
      println("Iteration "+k)

      val index = (k-1)%(numRDD)

      alpha = learningRate * (1 - wordCount/(trainWordsCount*numEpoch))
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + wordCount + ", alpha = " + alpha)

      val accWordCount = sc.accumulator(wordCount)

      println("index = "+index)

      //newSentences.sample(false, 1.0/numRDD).foreachPartition { iter =>
      val tmpRDD = sentenceSplit(index).mapPartitions { iter =>

        var sumIter = 0
        val hyperPara = bcHyperPara.value
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        val newIter = mutable.MutableList[Array[Int]]()

        while (iter.hasNext) {

          val sentence = iter.next()

          accWordCount += sentence.size

          var flag = false

          var sIter = 1

          while (sIter <= hyperPara.sentenceIter && (sIter == 1 || flag)) {

            for (posW <- 0 to sentence.size - 1) {

              var w = sentence(posW)

              //generate negative sampling (senses are generated randomly)
              val NEG = new Array[Int](hyperPara.negative)
              for (i <- 0 to hyperPara.negative - 1) {
                NEG(i) = hyperPara.table(Math.abs(util.Random.nextLong() % hyperPara.TABEL_SIZE).toInt)
                if (NEG(i) <= 0)
                  NEG(i) = (Math.abs(util.Random.nextLong()) % (hyperPara.vocabSize - 1) + 1).toInt
                NEG(i) = NEG(i) + hyperPara.vocabSize*util.Random.nextInt(hyperPara.numSenses)
              }

              //select the best sense
              sentence(posW) = sentence(posW)%vocabSize
              var bestSense = 0
              var bestScore = getScoreVariant(syn0,syn1,NEG,posW,sentence,hyperPara)
              for (sense <- 1 to numSenses - 1) {
                sentence(posW) = sentence(posW)%vocabSize + vocabSize*sense
                val tmpScore = getScoreVariant(syn0,syn1,NEG,posW,sentence,hyperPara)
                if (tmpScore > bestScore) {
                  bestScore = tmpScore
                  bestSense = sense
                }
              }
              sentence(posW) = sentence(posW)%vocabSize + vocabSize*bestSense

              //record if center word's sense is changed
              if (sentence(posW) != w)
                flag = true
              w = sentence(posW)

              //update syn1 (syn1 is for predicting surrounding word u)
              for (posU <- posW - hyperPara.window + 1 to posW + hyperPara.window - 1)
                if (posU >= 0 && posU < sentence.size && posU != posW) {
                  val u = sentence(posU)
                  val deltaU = getGradientULocalVariant(syn0, syn1, w, u, NEG, hyperPara)
                  blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaU, 1, syn1(u%hyperPara.vocabSize), 1)
                }

              //update syn0 (syn0 is for vectors of center word Z)
              var Z = w
              var label = 1
              for (d <- -1 to NEG.size - 1) {
                if (d >= 0) {
                  Z = NEG(d)
                  label = 0
                }
                val deltaZ = getGradientZLocalVariant(syn0, syn1, posW, Z, label, sentence, hyperPara)
                blas.saxpy(hyperPara.vectorSize, alpha.toFloat, deltaZ, 1, syn0(Z), 1)
              }

            }
            sIter+=1
          }
          sumIter += sIter
          newIter+=sentence
        }
        println("newIter.size="+newIter.size+" sumSentenceIter="+sumIter)
        newIter.toIterator
      }.cache()

      println("tmpRDD.count()="+tmpRDD.count())

      sentenceSplit(index).unpersist()
      sentenceSplit(index) = tmpRDD

      println("syn0Global(0)(0)="+syn0Global(0)(0))

      wordCount = accWordCount.value
    }

    writeToFileWithSensesVariant(outputPath, syn0Global, syn1Global)
  }



  private def writeToFileWithSenses(outputPath: String, syn0: Array[Array[Float]], syn1: Array[Array[Float]]): Unit={
    val file1 = new PrintWriter(new File(outputPath+"/wordIndexSenses.txt"))
    val file2 = new PrintWriter(new File(outputPath+"/syn0Senses.txt"))
    val file3 = new PrintWriter(new File(outputPath+"/syn1Senses.txt"))
    val iter = vocabHash.toIterator
    while (iter.hasNext) {
      val tmp = iter.next()
      for (s<-0 to numSenses-1)
        file1.write((tmp._2+s*vocabSize)+" "+tmp._1+"_"+s+"\n")
    }
    for (w <- 0 to vocabSize-1) {
      for (s <- 0 to numSenses - 1) {
        for (i <- 0 to vectorSize - 1) {
          file2.write(syn0(w+vocabSize*s)(i) + " ")
          file3.write(syn1(w+vocabSize*s)(i) + " ")
        }
        file2.write("\n")
        file3.write("\n")
      }
    }
    file1.close()
    file2.close()
    file3.close()
  }

  private def writeToFileWithSensesVariant(outputPath: String, syn0: Array[Array[Float]], syn1: Array[Array[Float]]): Unit={
    val file1 = new PrintWriter(new File(outputPath+"/wordIndexSenses.txt"))
    val file2 = new PrintWriter(new File(outputPath+"/syn0Senses.txt"))
    val file3 = new PrintWriter(new File(outputPath+"/syn1SensesVariant.txt"))
    val iter = vocabHash.toIterator
    while (iter.hasNext) {
      val tmp = iter.next()
      for (s<-0 to numSenses-1)
        file1.write((tmp._2+s*vocabSize)+" "+tmp._1+"_"+s+"\n")
    }
    for (w <- 0 to vocabSize-1) {
      for (s <- 0 to numSenses - 1) {
        for (i <- 0 to vectorSize - 1) {
          file2.write(syn0(w+vocabSize*s)(i) + " ")
        }
        file2.write("\n")
        file3.write("\n")
      }
      for (i <- 0 to vectorSize-1)
        file3.write(syn1(w)(i) + " ")
    }
    file1.close()
    file2.close()
    file3.close()
  }

  private def writeToFile(outputPath: String, syn0: Array[Array[Float]], syn1: Array[Array[Float]]): Unit={
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
        file2.write(syn0(a)(b)+" ")
        file3.write(syn1(a)(b)+" ")
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
