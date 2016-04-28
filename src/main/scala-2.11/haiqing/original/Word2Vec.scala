package haiqing.original

import java.lang.{Iterable => JavaIterable}

import scala.collection.JavaConverters._
import scala.collection.mutable

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader}
import org.apache.spark.rdd._

/**
 *  Entry in vocabulary
 */
private case class VocabWord(
                              var word: String,
                              var cn: Int,
                              var point: Array[Int],
                              var code: Array[Int],
                              var codeLen: Int
                              )

/**
 * Word2Vec creates vector representation of words in a text corpus.
 * The algorithm first constructs a vocabulary from the corpus
 * and then learns vector representation of words in the vocabulary.
 * The vector representation can be used as features in
 * natural language processing and machine learning algorithms.
 *
 * We used skip-gram model in our implementation and hierarchical softmax
 * method to train the model. The variable names in the implementation
 * matches the original C implementation.
 *
 * For original C implementation, see https://code.google.com/p/word2vec/
 * For research papers, see
 * Efficient Estimation of Word Representations in Vector Space
 * and
 * Distributed Representations of Words and Phrases and their Compositionality.
 */

class Word2Vec extends Serializable {

  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 1
  private var seed = util.Random.nextLong()
  private var minCount = 5
  private var maxSentenceLength = 1000

  /**
   * Sets the maximum length (in words) of each sentence in the input data.
   * Any sentence longer than this threshold will be divided into chunks of
   * up to `maxSentenceLength` size (default: 1000)
   */

  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    require(maxSentenceLength > 0,
      s"Maximum length of sentences must be positive but got ${maxSentenceLength}")
    this.maxSentenceLength = maxSentenceLength
    this
  }

  /**
   * Sets vector size (default: 100).
   */

  def setVectorSize(vectorSize: Int): this.type = {
    require(vectorSize > 0,
      s"vector size must be positive but got ${vectorSize}")
    this.vectorSize = vectorSize
    this
  }

  /**
   * Sets initial learning rate (default: 0.025).
   */

  def setLearningRate(learningRate: Double): this.type = {
    require(learningRate > 0,
      s"Initial learning rate must be positive but got ${learningRate}")
    this.learningRate = learningRate
    this
  }

  /**
   * Sets number of partitions (default: 1). Use a small number for accuracy.
   */

  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0,
      s"Number of partitions must be positive but got ${numPartitions}")
    this.numPartitions = numPartitions
    this
  }

  /**
   * Sets number of iterations (default: 1), which should be smaller than or equal to number of
   * partitions.
   */

  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations >= 0,
      s"Number of iterations must be nonnegative but got ${numIterations}")
    this.numIterations = numIterations
    this
  }

  /**
   * Sets random seed (default: a random long integer).
   */

  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Sets the window of words (default: 5)
   */

  def setWindowSize(window: Int): this.type = {
    require(window > 0,
      s"Window of words must be positive but got ${window}")
    this.window = window
    this
  }

  /**
   * Sets minCount, the minimum number of times a token must appear to be included in the word2vec
   * model's vocabulary (default: 5).
   */

  def setMinCount(minCount: Int): this.type = {
    require(minCount >= 0,
      s"Minimum number of times must be nonnegative but got ${minCount}")
    this.minCount = minCount
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40

  /** context words from [-window, window] */
  private var window = 5

  private var trainWordsCount = 0L
  private var vocabSize = 0
  @transient private var vocab: Array[VocabWord] = null
  @transient private var vocabHash = mutable.HashMap.empty[String, Int]

  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)

    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(x => VocabWord(
        x._1,
        x._2,
        new Array[Int](MAX_CODE_LENGTH),
        new Array[Int](MAX_CODE_LENGTH),
        0))
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

  private def createBinaryTree(): Unit = {
    val count = new Array[Long](vocabSize * 2 + 1)
    val binary = new Array[Int](vocabSize * 2 + 1)
    val parentNode = new Array[Int](vocabSize * 2 + 1)
    val code = new Array[Int](MAX_CODE_LENGTH)
    val point = new Array[Int](MAX_CODE_LENGTH)
    var a = 0
    while (a < vocabSize) {
      count(a) = vocab(a).cn
      a += 1
    }
    while (a < 2 * vocabSize) {
      count(a) = 1e9.toInt
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)
        vocab(a).point(i - b) = point(b) - vocabSize
        b += 1
      }
      a += 1
    }
  }

  /**
   * Computes the vector representation of each word in vocabulary.
   * @param dataset an RDD of sentences,
   *                each sentence is expressed as an iterable collection of words
   * @return a Word2VecModel
   */

  def fit[S <: Iterable[String]](dataset: RDD[S]): Word2VecModel = {

    learnVocab(dataset)

    createBinaryTree()

    val sc = dataset.context

    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)
    // each partition is a collection of sentences,
    // will be translated into arrays of Index integer
    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      // Each sentence will map to 0 or more Array[Int]
      sentenceIter.flatMap { sentence =>
        // Sentence of words, some of which map to a word index
        val wordIndexes = sentence.flatMap(bcVocabHash.value.get)
        // break wordIndexes into trunks of maxSentenceLength when has more
        wordIndexes.grouped(maxSentenceLength).map(_.toArray)
      }
    }
    
    val newSentences = sentences.repartition(numPartitions).cache()

    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue`.")
    }

    val syn0Global =
      Array.fill[Float](vocabSize * vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    var alpha = learningRate

    val sentenceSplited = newSentences.randomSplit(Array[Double](0.9,0.1), util.Random.nextLong())
    val trainSentence = sentenceSplited(0)
    val validationSentence = sentenceSplited(1)

    for (k <- 1 to numIterations) {
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)

      var loss = sc.accumulator(0.0)
      var lossNum = sc.accumulator(0)
      println("iteration = " + k + " adjust sense assignment and calculate loss for validation set...")
      validationSentence.foreachPartition { iter =>
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value
        iter.foldLeft(()) {
          case ((),sentence) =>
            lossNum += sentence.length
            var pos = 0
            while (pos < sentence.length) {
              val word = sentence(pos)
              val b = util.Random.nextInt(window)
              // Train Skip-gram
              var a = b
              while (a < window * 2 + 1 - b) {
                if (a != window) {
                  val c = pos - window + a
                  if (c >= 0 && c < sentence.length) {
                    val lastWord = sentence(c)
                    val l1 = lastWord * vectorSize
                    // Hierarchical softmax
                    var d = 0
                    while (d < bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(d)
                      val l2 = inner * vectorSize
                      // Propagate hidden -> output
                      var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                      if (f > -MAX_EXP && f < MAX_EXP) {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        f = expTable.value(ind)
                        if (bcVocab.value(word).code(d) == 1)
                          loss += -Math.log(1-f)
                        else
                          loss += -Math.log(f)
                        //println("f="+f)
                      }
                      d += 1
                    }
                  }
                }
                a += 1
              }
              pos += 1
            }
        }
      }
      println("Average loss per word: "+loss.value/lossNum.value+" "+loss.value+" "+lossNum.value)

      loss = sc.accumulator(0.0)
      lossNum = sc.accumulator(0)
      println("iteration = " + k + " adjust sense assignment and calculate loss for training set...")
      trainSentence.foreachPartition { iter =>
        val syn0 = bcSyn0Global.value
        val syn1 = bcSyn1Global.value
        iter.foldLeft(()) {
          case ((),sentence) =>
            lossNum += sentence.length
            var pos = 0
            while (pos < sentence.length) {
              val word = sentence(pos)
              val b = util.Random.nextInt(window)
              // Train Skip-gram
              var a = b
              while (a < window * 2 + 1 - b) {
                if (a != window) {
                  val c = pos - window + a
                  if (c >= 0 && c < sentence.length) {
                    val lastWord = sentence(c)
                    val l1 = lastWord * vectorSize
                    // Hierarchical softmax
                    var d = 0
                    while (d < bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(d)
                      val l2 = inner * vectorSize
                      // Propagate hidden -> output
                      var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                      if (f > -MAX_EXP && f < MAX_EXP) {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        f = expTable.value(ind)
                        if (bcVocab.value(word).code(d) == 1)
                          loss += -Math.log(1-f)
                        else
                          loss += -Math.log(f)
                        //println("f="+f)
                      }
                      d += 1
                    }
                  }
                }
                a += 1
              }
              pos += 1
            }
        }
      }
      println("Average loss per word: "+loss.value/lossNum.value+" "+loss.value+" "+lossNum.value)

      println("learn parameters and embeddings from training set ... ")
      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
        val syn0Modify = new Array[Int](vocabSize)
        val syn1Modify = new Array[Int](vocabSize)
        val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, 0L, 0L)) {
          case ((syn0, syn1, lastWordCount, wordCount), sentence) =>
            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              // TODO: discount by iteration?
              alpha =
                learningRate * (1 - (numPartitions*wordCount.toDouble+trainWordsCount*(k-1)) / (trainWordsCount*numIterations + 1))
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
              //println("idx = "+idx+"  ,  alpha = " + alpha)
            }
            wc += sentence.length
            var pos = 0
            while (pos < sentence.length) {
              val word = sentence(pos)
              val b = util.Random.nextInt(window)
              // Train Skip-gram
              var a = b
              while (a < window * 2 + 1 - b) {
                if (a != window) {
                  val c = pos - window + a
                  if (c >= 0 && c < sentence.length) {
                    val lastWord = sentence(c)
                    val l1 = lastWord * vectorSize
                    val neu1e = new Array[Float](vectorSize)
                    // Hierarchical softmax
                    var d = 0
                    while (d < bcVocab.value(word).codeLen) {
                      val inner = bcVocab.value(word).point(d)
                      val l2 = inner * vectorSize
                      // Propagate hidden -> output
                      var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                      if (f > -MAX_EXP && f < MAX_EXP) {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        f = expTable.value(ind)
                        val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                        blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                        blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                        syn1Modify(inner) += 1
                      }
                      d += 1
                    }
                    blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                    syn0Modify(lastWord) += 1
                  }
                }
                a += 1
              }
              pos += 1
            }
            (syn0, syn1, lwc, wc)
        }
        val syn0Local = model._1
        val syn1Local = model._2
        // Only output modified vectors.
        Iterator.tabulate(vocabSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten
      }
      val synAgg = partial.reduceByKey { case (v1, v2) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        v1
      }.collect()
      var i = 0
      while (i < synAgg.length) {
        val index = synAgg(i)._1
        if (index < vocabSize) {
          val newArr = new Array[Float](vectorSize)
          blas.saxpy(vectorSize, 1.0f/numPartitions, synAgg(i)._2, 1, newArr, 1)
          Array.copy(newArr, 0, syn0Global, index * vectorSize, vectorSize)
        } else {
          val newArr = new Array[Float](vectorSize)
          blas.saxpy(vectorSize, 1.0f/numPartitions, synAgg(i)._2, 1, newArr, 1)
          Array.copy(newArr, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
        }
        i += 1
      }
      bcSyn0Global.unpersist(false)
      bcSyn1Global.unpersist(false)
    }
    newSentences.unpersist()

    val wordArray = vocab.map(_.word)
    new Word2VecModel(wordArray.zipWithIndex.toMap, syn0Global)
  }

  /**
   * Computes the vector representation of each word in vocabulary (Java version).
   * @param dataset a JavaRDD of words
   * @return a Word2VecModel
   */
  def fit[S <: JavaIterable[String]](dataset: JavaRDD[S]): Word2VecModel = {
    fit(dataset.rdd.map(_.asScala))
  }
}

/**
 * Word2Vec model
 * @param wordIndex maps each word to an index, which can retrieve the corresponding
 *                  vector from wordVectors
 * @param wordVectors array of length numWords * vectorSize, vector corresponding
 *                    to the word mapped with index i can be retrieved by the slice
 *                    (i * vectorSize, i * vectorSize + vectorSize)
 */
class Word2VecModel (
                      val wordIndex: Map[String, Int],
                      val wordVectors: Array[Float]) extends Serializable  {

  private val numWords = wordIndex.size
  // vectorSize: Dimension of each word's vector.
  private val vectorSize = wordVectors.length / numWords

  // wordList: Ordered list of words obtained from wordIndex.
  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  // wordVecNorms: Array of length numWords, each value being the Euclidean norm
  //               of the wordVector.
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

  /**
   * Transforms a word to its vector representation
   * @param word a word
   * @return vector representation of word
   */
  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  /**
   * Find synonyms of a word
   * @param word a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  /**
   * Find synonyms of the vector representation of a word
   * @param vector vector representation of a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
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
    val vecNorm = blas.snrm2(vectorSize, fVector, 1)
    while (ind < numWords) {
      val norm = wordVecNorms(ind)
      if (norm == 0.0) {
        cosVec(ind) = 0.0
      } else {
        cosVec(ind) /= norm
      }
      ind += 1
    }
    var topResults = wordList.zip(cosVec)
      .toSeq
      .sortBy(-_._2)
      .take(num + 1)
      .tail
    if (vecNorm != 0.0f) {
      topResults = topResults.map { case (word, cosVal) =>
        (word, cosVal / vecNorm)
      }
    }
    topResults.toArray
  }

  /**
   * Returns a map of words to their vector representations.
   */
  def getVectors: Map[String, Array[Float]] = {
    wordIndex.map { case (word, ind) =>
      (word, wordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize))
    }
  }

}
