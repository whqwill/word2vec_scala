package haiqing.word2vec

import java.io.{PrintWriter, File}

import com.github.fommil.netlib.BLAS._
import org.apache.spark.AccumulatorParam
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseMatrix, BLAS, DenseVector}

import scala.io.Source

/**
 * Created by hwang on 17.11.15.
 */

case class VocabWord(
                      var word: String,
                      var cn: Int
                      )

object ArrayAccumulatorParam extends AccumulatorParam[Array[Float]] {
  def zero(initialValue: Array[Float]): Array[Float] = {
    new Array[Float](initialValue.size)
  }
  def addInPlace(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    for(a<-0 to v1.size-1)
      v1(a) += v2(a)
    v1
  }
}

class MSSkipGram extends Serializable{

  private var skipgram: SkipGram = null
  private var path: String = "./"
  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 100
  private var seed = util.Random.nextLong()
  private var minCount = 5
  private var negative = 5
  private var numSenses = 3
  private var adjustingRatio = 0.5
  private var window = 5
  private var sample = 0.01
  private var sentenceIter = 5
  private var testWord : String = null
  private var printRadio = 0.1
  private var saveRadio = 0.1
  def setPrintRadio(printRadio: Double): this.type = {
    this.printRadio = printRadio
    this
  }
  def setSaveRadio(saveRadio: Double): this.type = {
    this.saveRadio = saveRadio
    this
  }
  def setTestWord(testWord: String): this.type = {
    this.testWord = testWord
    this
  }
  def setSkipGram(skipgram: SkipGram): this.type = {
    this.skipgram = skipgram
    this
  }
  def setPath(path: String): this.type = {
    this.path = path
    this
  }
  def setAdjustingRatio(adjustingRatio: Double): this.type = {
    this.adjustingRatio = adjustingRatio
    this
  }
  def setSentenceIter(sentenceIter: Int): this.type = {
    this.sentenceIter = sentenceIter
    this
  }
  def setSample(sample: Double): this.type = {
    this.sample = sample
    this
  }
  def setWindow(window: Int): this.type = {
    this.window = window
    this
  }
  def setNumSenses(numSenses: Int): this.type = {
    this.numSenses = numSenses
    this
  }
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
  def setNegative(negative: Int): this.type = {
    this.negative = negative
    this
  }
  def setMinCount(minCount: Int): this.type = {
    this.minCount = minCount
    this
  }
  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_SENTENCE_LENGTH = 1000
  private val POWER = 0.75
  private val VARIANCE = 0.01f
  private val TABEL_SIZE = 10000
  private var trainWordsCount = 0
  private var vocabSize = 0
  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var syn0Global: Array[Float] = null
  private var syn1Global: Array[Float] = null
  private var syn0GlobalMS: Array[Float] = null
  private var syn1GlobalMS: Array[Float] = null

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

  private def activeFunction(syn0 :Array[Float], syn1 :Array[Float], l0: Int, l1: Int, expTable: Array[Float], vectorSize: Int, MAX_EXP: Int): Double = {
    var f = blas.sdot(vectorSize, syn0, l0, 1, syn1, l1, 1)
    if (f > MAX_EXP)
      f = expTable(expTable.length - 1)
    else if (f < -MAX_EXP)
      f = expTable(0)
    else {
      val ind = ((f + MAX_EXP) * (expTable.size / MAX_EXP / 2.0)).toInt
      f = expTable(ind)
    }
    f
  }

  private def getScore(syn0 :Array[Float], syn1 :Array[Float], NEG: Array[Int], posW: Int, sentence: Array[Int], expTable: Array[Float], window: Int, vectorSize: Int, vocabSize: Int, MAX_EXP: Int): Double = {
    var score = 0.0
    val w = sentence(posW)
    for (posU <- posW-window+1 to posW+window-1) {
      if (posU >= 0 && posU < sentence.size && posW != posW) {
          val u = sentence(posU)
          val l0 = u * vectorSize
          val l1 = w * vectorSize
          score += math.log(activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP))

          for (d <- 0 to NEG.size-1) {
              val Z = NEG(d)
              val l1 = Z * vectorSize
              score += math.log(1-activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP))
          }
      }
    }
    score
  }

  private def getGradientU(syn0 :Array[Float], syn1 :Array[Float],  w: Int, u: Int, NEG: Array[Int], expTable: Array[Float], vectorSize: Int, vocabSize: Int, MAX_EXP: Int): Array[Float] = {
        val Delta = new Array[Float](vectorSize)

          val l0 = u * vectorSize
          var Z = w
          var label = 1

          for (d <- -1 to NEG.size-1) {
              if (d >= 0) {
                  Z = NEG(d)
                  label = 0
              }

              val l1 = Z * vectorSize
              val g = (label-activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP)).toFloat

              blas.saxpy(vectorSize, g, syn0, l0, 1, Delta, 0, 1)
          }
        Delta
  }

  private def getGradientZ(syn0 :Array[Float], syn1 :Array[Float],  posW: Int, Z: Int, label: Int, window: Int, sentence: Array[Int], expTable: Array[Float], vectorSize: Int, vocabSize: Int, MAX_EXP: Int): Array[Float] = {
        val Delta = new Array[Float](vectorSize)
        for (posU <- posW-window+1 to posW+window-1) {
          if (posU >= 0 && posU < sentence.size && posW != posW) {
            val u = sentence(posU)
            val l0 = u * vectorSize
            val l1 = Z * vectorSize
            val g = (label-activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP)).toFloat

            blas.saxpy(vectorSize, g, syn1, l1, 1, Delta, 0, 1)
          }
        }
        Delta
  }

  def fit (words: RDD[String]): Word2VecModel = {

    var syn0 :Array[Float] = null
    var syn1 :Array[Float] = null

    if (skipgram != null) {
      trainWordsCount = skipgram.getTrainWordsCount()
      vocabSize = skipgram.getVocabSize()
      vocab = skipgram.getVocab()
      vocabHash = skipgram.getVocabHash()

      syn0 = skipgram.getSyn0()
      syn1 = skipgram.getSyn1()

      syn0Global = new Array[Float](vocabSize * vectorSize * numSenses)
      syn1Global = new Array[Float](vocabSize * vectorSize * numSenses)
      for (a <- 0 to syn0.size-1)
        for (i <- 0 to numSenses-1) { //it is a problem
          syn0Global(i*syn0.size+a) = syn0(a)+(util.Random.nextFloat()-0.5f)*VARIANCE
          syn1Global(i*syn0.size+a) = syn1(a)+(util.Random.nextFloat()-0.5f)*VARIANCE
        }

      println("Initialize lookup-table from Skip-Gram.")
    }
    else if (new java.io.File(path+"/wordVectors.txt").exists){
      learnVocab(words)
      syn0 = Source.fromFile(path+"/wordVectors.txt").getLines().next().split(" ").map(s=>s.toFloat)

      syn0Global = new Array[Float](vocabSize * vectorSize * numSenses)
      syn1Global = new Array[Float](vocabSize * vectorSize * numSenses)
      for (a <- 0 to syn0.size-1)
        for (i <- 0 to numSenses-1)
          syn0Global(i*syn0.size+a) = syn0(a)+(util.Random.nextFloat()-0.5f)*VARIANCE

      println("Initialize lookup-table from file.")
    }
    else {
      learnVocab(words)
      syn0Global = Array.fill[Float](vocabSize * vectorSize * numSenses)((util.Random.nextFloat() - 0.5f) / vectorSize)
      syn1Global = new Array[Float](vocabSize * vectorSize * numSenses)
      println("Initialize lookup-table randomly.")
    }


    println("trainWordsCount = " + trainWordsCount)

    val sc = words.context
    val expTable = sc.broadcast(createExpTable())
    val bcVocabHash = sc.broadcast(vocabHash)
    val table = sc.broadcast(makeTable())

    //sentence need to be changed
    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext
        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            word match {
              case Some(w) =>
                sentence += w+vocabSize*util.Random.nextInt(numSenses)
                sentenceLength += 1
              case None =>
            }
          }
          sentence.result()
        }
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()

    util.Random.setSeed(seed)
    if (vocabSize.toLong * vectorSize * 8 >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue/8`.")
    }

    val sentenceSplit = newSentences.randomSplit(new Array[Double]((1/sample).toInt).map(x=>x+1))


    var alpha = learningRate
    for (k <- 1 to numIterations) {
      println("Iteration "+k)

      /*
      if (k > 0 && k % (numIterations*printRadio).toInt == 0) {
        val wordArray = vocab.map(_.word)
        val msWordArray = new Array[String](wordArray.size*numSenses)
        for (a <- 0 to wordArray.size-1)
          for (i <- 0 to numSenses-1)
            msWordArray(i*wordArray.length+a) = wordArray(a)+i
        val model = new Word2VecModel(msWordArray.zipWithIndex.toMap, syn0Global)
        for (i <- 0 to numSenses-1) {
          val newSynonyms = model.findSynonyms(testWord+i, 20)
          println()
          for ((synonym, cosineSimilarity) <- newSynonyms) {
            println(s"$synonym $cosineSimilarity")
          }
        }
      }
*/

      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)

      alpha =
        learningRate * (1 - (k-1)*1.0/numIterations)
      //println("!!"+"numIterations"+numIterations+"numPartitions"+numPartitions+(trainWordsCount*k + numPartitions * wordCount.toDouble) / (trainWordsCount + 1) / numIterations)
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + sample*(k-1)*trainWordsCount + ", alpha = " + alpha)


      val syn0Accum = sc.accumulator(new Array[Float](vocabSize * vectorSize * numSenses))(ArrayAccumulatorParam)
      val syn1Accum = sc.accumulator(new Array[Float](vocabSize * vectorSize * numSenses))(ArrayAccumulatorParam)

      sentenceSplit((k-1)%(1/sample).toInt) = sentenceSplit((k-1)%(1/sample).toInt).mapPartitions { iter =>

        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        val newSentenseIter = mutable.MutableList[Array[Int]]()

        while (iter.hasNext) {
          val sentence = iter.next()

          for (sIter <- 1 to sentenceIter) {
            var pos = 0
            while (pos < sentence.size) {
              var word = sentence(pos)

              val b = util.Random.nextInt(window)
              //negative sampling
              val negSample = new Array[Int](negative)
              for (i <- 0 to negative - 1) {
                negSample(i) = table.value(Math.abs(util.Random.nextLong() % TABEL_SIZE).toInt)
                if (negSample(i) <= 0)
                  negSample(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt
                negSample(i) = negSample(i) + util.Random.nextInt(numSenses) * vocabSize
              }

              if (k <= numIterations * adjustingRatio) {

                //adjust the senses
                var bestSense = -1
                var bestScore = 0.0

                for (sense <- 0 to numSenses - 1) {
                  var a = b
                  var score = 1.0
                  word = word % vocabSize + vocabSize * sense

                  if (bestSense == -1 || score > bestScore) {
                    bestScore = score
                    bestSense = sense
                  }
                }
                word = word % vocabSize + vocabSize * bestSense
                sentence(pos) = word

                //train Skip-Gram
                var a = b
                while (a < window * 2 + 1 - b) {
                  if (a != window) {
                    val c = pos - window + a
                    if (c >= 0 && c < sentence.size) {
                      val lastWord = sentence(c)
                      val l1 = lastWord * vectorSize
                      val neu1e = new Array[Float](vectorSize)
                      var target = word
                      var label = 1
                      for (d <- 0 to negative) {
                        if (d > 0) {
                          target = negSample(d - 1)
                          label = 1
                        }
                        if (target % vocabSize != lastWord % vocabSize || d == 0) {
                          val l2 = target * vectorSize
                          // Propagate hidden -> output
                          var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                          var g = 0.0f
                          if (f > MAX_EXP)
                            g = (label - 1) * alpha.toFloat
                          else if (f < -MAX_EXP)
                            g = (label - 0) * alpha.toFloat
                          else {
                            val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                            f = expTable.value(ind)
                            g = (label - f) * alpha.toFloat
                          }
                          blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                          blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                        }
                      }
                      blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                    }
                  }
                  a += 1
                }

              }
              pos += 1
            }
          }

          newSentenseIter+=sentence
        }

        syn0Accum += syn0
        syn1Accum += syn1

        newSentenseIter.toIterator
      }

      sentenceSplit((k-1)%(1/sample).toInt).count()

      syn0Global = syn0Accum.value
      syn1Global = syn1Accum.value

      for (a <- 0 to syn0Global.size-1) {
        syn0Global(a) /= numPartitions
        syn1Global(a) /= numPartitions
      }
      bcSyn0Global.unpersist(false)
      bcSyn1Global.unpersist(false)
    }
    newSentences.unpersist()

    val wordArray = vocab.map(_.word)
    val msWordArray = new Array[String](wordArray.size*numSenses)
    for (a <- 0 to wordArray.size-1)
      for (i <- 0 to numSenses-1)
        msWordArray(i*wordArray.length+a) = wordArray(a)+i
    new Word2VecModel(msWordArray.zipWithIndex.toMap, syn0Global)

  }
}

class SkipGram extends Serializable {
  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 100
  private var seed = util.Random.nextLong()
  private var minCount = 5
  private var negative = 5
  private var window = 5
  private var sample = 0.01
  private var subSampling = 0.0001
  private var testWord : String = null
  private var printRadio = 0.1
  private var saveRadio = 0.1
  def setPrintRadio(printRadio: Double): this.type = {
    this.printRadio = printRadio
    this
  }
  def setSaveRadio(saveRadio: Double): this.type = {
    this.saveRadio = saveRadio
    this
  }
  def setTestWord(testWord: String): this.type = {
    this.testWord = testWord
    this
  }
  def setSample(sample: Double): this.type = {
    this.sample = sample
    this
  }
  def setWindow(window: Int): this.type = {
    this.window = window
    this
  }
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

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_SENTENCE_LENGTH = 1000
  private val POWER = 0.75
  private val TABEL_SIZE = 10000
  private var trainWordsCount = 0
  private var vocabSize = 0
  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var syn0Global: Array[Float] = null
  private var syn1Global: Array[Float] = null

  def cleanSyn(): Unit={
    syn0Global = Array.fill[Float](vocabSize * vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
    syn1Global = new Array[Float](vocabSize * vectorSize)
  }

  def getVocabSize() = vocabSize
  def getVocab() = vocab
  def getVocabHash() = vocabHash
  def getSyn0() = syn0Global
  def getSyn1() = syn1Global
  def getTrainWordsCount() = trainWordsCount

  private def learnVocab(words: RDD[String]): Unit = {
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .map(x => VocabWord(
        x._1,
        x._2))
      .filter(_.cn >= minCount)
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    for (a <- vocab.toIterator)
      println(a)
    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    println("trainWordsCount = " + trainWordsCount)
  }

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
    var trainWordsPow = 0.0;
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


  def load (words: RDD[String], model: Word2VecModel): Unit={
    learnVocab(words)

  }

  private def activeFunction(syn0 :Array[Float], syn1 :Array[Float], l0: Int, l1: Int, expTable: Array[Float], vectorSize: Int, MAX_EXP: Int): Double = {
    var f = blas.sdot(vectorSize, syn0, l0, 1, syn1, l1, 1)
    if (f > MAX_EXP)
      f = expTable(expTable.length - 1)
    else if (f < -MAX_EXP)
      f = expTable(0)
    else {
      val ind = ((f + MAX_EXP) * (expTable.size / MAX_EXP / 2.0)).toInt
      f = expTable(ind)
    }
    f
  }

  private def getScore(syn0 :Array[Float], syn1 :Array[Float], NEG: Array[Int], posW: Int, sentence: Array[Int], expTable: Array[Float], window: Int, vectorSize: Int, vocabSize: Int, MAX_EXP: Int): Double = {
    var score = 0.0
    val w = sentence(posW)
    for (posU <- posW-window+1 to posW+window-1) {
      if (posU >= 0 && posU < sentence.size && posW != posW) {
        val u = sentence(posU)
        val l1 = u * vectorSize
        val l0 = w * vectorSize
        score += math.log(activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP))

        for (d <- 0 to NEG.size-1) {
          val Z = NEG(d)
          val l0 = Z * vectorSize
          score += math.log(1-activeFunction(syn0, syn1, l0, l1, expTable, vectorSize, MAX_EXP))
        }
      }
    }
    score
  }

  private def getGradientU(syn0 :Array[Float], syn1 :Array[Float],  w: Int, u: Int, NEG: Array[Int], expTable: Array[Float], vectorSize: Int, MAX_EXP: Int): Array[Float] = {
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

  def fit (words: RDD[String]): Word2VecModel = {
    learnVocab(words)

    println("vocabSize: "+vocabSize)
    val sc = words.context

    val expTable = sc.broadcast(createExpTable())
    val bcVocabHash = sc.broadcast(vocabHash)
    val table = sc.broadcast(makeTable())

    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext

        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            // The subsampling randomly discards frequent words while keeping the ranking same
            //if (word.nonEmpty && scala.math.sqrt((subSampling*trainWordsCount)/bcVocab.value(word.get).cn) > util.Random.nextDouble()) {
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

    /*
    if (vocabSize.toLong * vectorSize * 8 >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue/8`.")
    }*/

    var alpha = learningRate
    println("Debug vocab size = " )
    syn0Global = Array.fill[Float](vocabSize * vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
    syn1Global = new Array[Float](vocabSize * vectorSize)

    val sentenceSplit = newSentences.randomSplit(new Array[Double]((1/sample).toInt).map(x=>x+1))

    for (k <- 1 to numIterations) {

      println("Iteration "+k)

      /*if (k > 0 && k % (numIterations*printRadio).toInt == 0) {
        val model = new Word2VecModel(vocab.map(_.word).zipWithIndex.toMap, syn0Global)
        val newSynonyms = model.findSynonyms(testWord, 20)
        println()
        for ((synonym, cosineSimilarity) <- newSynonyms) {
          println(s"$synonym $cosineSimilarity")
        }
      }*/

      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val bcSeed = sc.broadcast(k*seed)

      alpha =
        learningRate * (1 - (k-1)*1.0/numIterations)
      //println("!!"+"numIterations"+numIterations+"numPartitions"+numPartitions+(trainWordsCount*k + numPartitions * wordCount.toDouble) / (trainWordsCount + 1) / numIterations)
      if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
      println("wordCount = " + sample*(k-1)*trainWordsCount + ", alpha = " + alpha)

      val syn0Accum = sc.accumulator(new Array[Float](vocabSize * vectorSize))(ArrayAccumulatorParam)
      val syn1Accum = sc.accumulator(new Array[Float](vocabSize * vectorSize))(ArrayAccumulatorParam)

      sentenceSplit((k-1)%((1/sample).toInt)) = sentenceSplit((k-1)%((1/sample).toInt)).mapPartitionsWithIndex { (idx, iter) =>

        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value

        util.Random.setSeed(bcSeed.value+idx)

        val newSentenseIter = mutable.MutableList[Array[Int]]()

        while (iter.hasNext) {

          val sentence = iter.next()

          for (posW <- 0 to sentence.size-1) {

            val w = sentence(posW)
            val NEG = new Array[Int](negative)
            for (i <- 0 to negative - 1) {
              NEG(i) = table.value(Math.abs(util.Random.nextLong() % TABEL_SIZE).toInt)
              if (NEG(i) <= 0)
                NEG(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt
            }

            for (posU <- posW-window+1 to posW+window-1)
              if (posU >= 0 && posU < sentence.size && posU != posW) {
                  val u = sentence(posU)
                  val l1 = u * vectorSize

                  val deltaU = getGradientU(syn0,syn1,w,u,NEG,expTable.value,vectorSize,MAX_EXP)

                  blas.saxpy(vectorSize, alpha.toFloat, deltaU, 0, 1, syn1, l1, 1)
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
          newSentenseIter+=sentence
        }

        syn0Accum += syn0
        syn1Accum += syn1

        newSentenseIter.toIterator
      }
      sentenceSplit((k-1)%((1/sample).toInt)).count()


      syn0Global = syn0Accum.value
      syn1Global = syn1Accum.value

      for (a <- 0 to syn0Global.size-1) {
        syn0Global(a) /= numPartitions
        syn1Global(a) /= numPartitions
      }
      bcSyn0Global.unpersist(false)
      bcSyn1Global.unpersist(false)
    }
    newSentences.unpersist()

    val wordArray = vocab.map(_.word)
    new Word2VecModel(wordArray.zipWithIndex.toMap, syn0Global)
  }
}


class Word2VecModel (
                      private val wordIndex: Map[String, Int],
                      private val wordVectors: Array[Float]){


  private val numWords = wordIndex.size
  private val vectorSize = wordVectors.length / numWords

  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  private val wordVecNorms: Array[Double] = {
    val wordVecNorms = new Array[Double](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
      i += 1
    }
    wordVecNorms
  }

  private def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / norm1 / norm2
  }

  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = Array.fill[Float](numWords)(0)
    val alpha: Float = 1
    val beta: Float = 0

    blas.sgemv(
      "T", vectorSize, numWords, alpha, wordVectors, vectorSize, fVector, 1, beta, cosineVec, 1)

    // Need not divide with the norm of the given vector since it is constant.
    val cosVec = cosineVec.map(_.toDouble)
    var ind = 0
    while (ind < numWords) {
      cosVec(ind) /= wordVecNorms(ind)
      ind += 1
    }
    wordList.zip(cosVec)
      .toSeq
      .sortBy(- _._2)
      .take(num + 1)
      .tail
      .toArray
  }

  def getVectors: Map[String, Array[Float]] = {
    wordIndex.map { case (word, ind) =>
      (word, wordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize))
    }
  }

  def save(path: String): Unit = {
    val file1 = new PrintWriter(new File(path+"/wordIndex.txt"))
    val file2 = new PrintWriter(new File(path+"/wordVectors.txt"))
    val iter = wordIndex.toIterator
    while (iter.hasNext) {
      val tmp = iter.next()
      file1.write(tmp._1+" "+tmp._2+"\n")
    }
    for (i <- 0 to wordVectors.size-2)
      file2.write(wordVectors(i)+" ")
    file2.write(wordVectors(wordVectors.size-1)+"\n")
    file1.close()
    file2.close()
  }
}

