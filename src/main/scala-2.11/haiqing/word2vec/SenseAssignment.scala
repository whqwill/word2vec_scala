package haiqing.word2vec

import java.io._

import com.github.fommil.netlib.BLAS._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder
import scala.io.Source
import com.github.fommil.netlib.BLAS.{getInstance => blas}
/**
 * Created by hwang on 09.02.16.
 */
class SenseAssignment extends Serializable {

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val POWER = 0.75
  private val VARIANCE = 0.01f
  private val TABEL_SIZE = 10000

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

  private def makeNegativaSampleTable(): Array[Int] = {
    val table = new Array[Int](TABEL_SIZE)
    var trainWordsPow = 0.0
    for (a <- 0 to vocabSize - 1)
      trainWordsPow += Math.pow(vocab(a).cn, POWER)
    var i = 0
    var d1 = Math.pow(vocab(i).cn, POWER) / trainWordsPow
    for (a <- 0 to TABEL_SIZE - 1) {
      table(a) = i
      if (a * 1.0 / TABEL_SIZE > d1) {
        i += 1
        d1 += Math.pow(vocab(i).cn, POWER) / trainWordsPow
      }
      if (i >= vocabSize)
        i = vocabSize - 1
    }
    table
  }


  private var vocabSize = 0
  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var senseTable: Array[Int] = null
  private var words: RDD[String] = null
  private var totalWords = 0
  private var sentences: RDD[Array[Int]] = null
  private var totalSentences = 0
  private var numRDDs = 0
  private var sentencesSplit: Array[RDD[Array[Int]]] = null
  private var syn0Sense: Array[Array[Array[Float]]] = null
  private var syn1Sense: Array[Array[Array[Float]]] = null
  private var vectorSize = 0
  private var learned = false

  def loadData(input: RDD[String]): Unit = {
    words = input.map(line => line.split(" ").toSeq).flatMap(x => x).cache()
    words.count()
  }

  def preprocessData(): Unit = {
    require(words != null, "words RDD is null. You may need to check if loading data correctly.")
    words = words.map(x => x.toLowerCase).filter(x => x.size > 0).filter(x => x.size > 1 || x(0).isLetter || x(0).isDigit).cache()
    words.count()
    println("words.count()=" + words.count())
  }

  def learnVocab(minCount: Int = 5): Unit = {
    require(words != null, "words RDD is null. You may need to check if loading data correctly.d")
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

    for (a <- 0 to vocabSize - 1) {
      vocabHash += vocab(a).word -> a
      totalWords += vocab(a).cn
      if (vocab(a).word == "day")
        println(vocab(a).toString)
    }

    var multiSenseWords = 0
    senseTable = new Array[Int](vocabSize)

    println("print some words in vocabulary: ")
    for (a <- 0 to vocabSize - 1) {
      if (vocab(a).cn > totalWords / vocabSize /10) {
        senseTable(a) = 2
        multiSenseWords += 1
      }
      else
        senseTable(a) = 1
      if (util.Random.nextInt(vocabSize) < 30)
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
    }

    println("multiSenseWords = " + multiSenseWords)
    println("vocabSize = " + vocabSize)
    println("totalWords = " + totalWords)
  }

  def learnVocabWithoutSense(minCount: Int = 5): Unit = {
    require(words != null, "words RDD is null. You may need to check if loading data correctly.d")
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

    for (a <- 0 to vocabSize - 1) {
      vocabHash += vocab(a).word -> a
      totalWords += vocab(a).cn

    }

    senseTable = new Array[Int](vocabSize)

    println("print some words in vocabulary: ")
    for (a <- 0 to vocabSize - 1) {
      senseTable(a) = 1
      if (util.Random.nextInt(vocabSize) < 30)
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
    }

    println("vocabSize = " + vocabSize)
    println("totalWords = " + totalWords)
  }

  def makeSentences(maxSentenceLength: Int = 3000): Unit = {
    require(words != null, "words RDD is null. You may need to check if loading data correctly.")
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check if learning vocabulary correctly.")



    val sc = words.context
    val bcVacabHash = sc.broadcast(vocabHash)
    val bcNumSensesTable = sc.broadcast(senseTable)
    val bcVocabSize = sc.broadcast(vocabSize)

    sentences = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext

        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < maxSentenceLength) {
            val word = bcVacabHash.value.get(iter.next())
            if (word.nonEmpty) {
              sentence += word.get + bcVocabSize.value * util.Random.nextInt(bcNumSensesTable.value(word.get))
              sentenceLength += 1
            }
          }
          sentence.result()
        }
      }
    }.cache()

    totalSentences = sentences.count().toInt
    println("totalSentences = " + totalSentences)
  }

  def splitRDD(numSentencesPerIterPerCore: Int = 5): Unit = {
    require(totalSentences > 0, "The number of sentences should be > 0. You may need to check if making sentences correctly.")

    val sc = words.context
    val numCores = sc.defaultParallelism
    val numSentencesPerIter = numSentencesPerIterPerCore * numCores
    numRDDs = totalSentences / numSentencesPerIter
    sentencesSplit = sentences.randomSplit(new Array[Double](numRDDs).map(x => x + 1))

    println("numSentencesPerIter = " + numSentencesPerIter)
    println("numRDDs = " + numRDDs)
  }

  //initialize from normal skip-gram model
  def initializeParameters(synPath: String, vectorSize: Int): Unit = {


    val syn0 = Source.fromFile(synPath + "/syn0Sense.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s => s.toFloat).toArray
    val syn1 = Source.fromFile(synPath + "/syn1Sense.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s => s.toFloat).toArray

    require(vectorSize * vocabSize == syn0.size, "syn0.size should be equal to vectorSize*vocabSize. You may need to check if vectorSize or vocabSize or syn0 file is correct.")
    require(vectorSize * vocabSize == syn1.size, "syn1.size should be equal to vectorSize*vocabSize. You may need to check if vectorSize or vocabSize or syn1 file is correct.")

    syn0Sense = new Array[Array[Array[Float]]](vocabSize)
    syn1Sense = new Array[Array[Array[Float]]](vocabSize)

    for (word <- 0 to vocabSize - 1) {

      syn0Sense(word) = new Array[Array[Float]](senseTable(word))
      syn1Sense(word) = new Array[Array[Float]](senseTable(word))


      for (sense <- 0 to senseTable(word) - 1) {

        syn0Sense(word)(sense) = new Array[Float](vectorSize)
        syn1Sense(word)(sense) = new Array[Float](vectorSize)
        for (i <- 0 to vectorSize - 1) {
          syn0Sense(word)(sense)(i) = syn0(word * vectorSize + i) + (util.Random.nextFloat() - 0.5f) * VARIANCE
          syn1Sense(word)(sense)(i) = syn1(word * vectorSize + i) + (util.Random.nextFloat() - 0.5f) * VARIANCE
        }
      }
    }

    this.vectorSize = vectorSize
  }

  //initialize randomly
  def initializeParameters(vectorSize: Int = 100): Unit = {

    util.Random.setSeed(42)

    syn0Sense = new Array[Array[Array[Float]]](vocabSize)
    syn1Sense = new Array[Array[Array[Float]]](vocabSize)

    for (word <- 0 to vocabSize - 1) {

      syn0Sense(word) = new Array[Array[Float]](senseTable(word))
      syn1Sense(word) = new Array[Array[Float]](senseTable(word))

      for (sense <- 0 to senseTable(word) - 1) {

        syn0Sense(word)(sense) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
        syn1Sense(word)(sense) = new Array[Float](vectorSize)
      }
    }

    this.vectorSize = vectorSize
  }

  def train_local(numEpochs: Int = 2, window: Int = 5, numNeg: Int = 5, learningRate: Float = 0.025f, seed: Long = util.Random.nextLong()): Unit = {
    require(sentencesSplit.size > 0, "The number of sentences RDD should be > 0. You may need to check if splitting RDD correctly.")
    require(syn0Sense != null, "syn0Sense should not be null. You may need to check if initializing parameters correctly.")
    require(syn1Sense != null, "syn1Sense should not be null. You may need to check if initializing parameters correctly.")

    val sc = sentencesSplit(0).context

    val totalInterations = numEpochs * numRDDs

    val bcSyn0Sense = sc.broadcast(syn0Sense)
    val bcSyn1Sense = sc.broadcast(syn1Sense)

    val bcExpTable = sc.broadcast(createExpTable())
    val bcNegTable = sc.broadcast(makeNegativaSampleTable())
    val bcSenseTable = sc.broadcast(senseTable)

    val hyperPara = mutable.HashMap[String, AnyVal]()
    hyperPara.put("window", window)
    hyperPara.put("numNeg", numNeg)
    hyperPara.put("learningRate", learningRate)
    hyperPara.put("seed", seed)
    hyperPara.put("vocabSize", vocabSize)
    hyperPara.put("vectorSize", vectorSize)
    hyperPara.put("MAX_EXP", MAX_EXP)

    for (iteration <- 1 to totalInterations) {

      var alpha = learningRate * (1 - (iteration-1) * 1.0f / totalInterations)
      if (alpha < learningRate * 0.0001f)
        alpha = learningRate * 0.0001f

      hyperPara.put("iteration", iteration)
      hyperPara.put("alpha", alpha)
      val bcHyperPara = sc.broadcast(hyperPara)

      val indexRDD = (iteration-1) % numRDDs

      println("iteration = " + iteration + "   indexRDD = " + indexRDD + "   alpha = " + alpha)


      sentencesSplit(indexRDD) = sentencesSplit(indexRDD).mapPartitions { sentenceIter =>

        //println("mapPartitions ... ... ...")
        val newSentenceIter = mutable.MutableList[Array[Int]]()
        for (sentence <- sentenceIter) {
          val newSentence = adjustAssignment(sentence, bcHyperPara.value, bcExpTable.value, bcNegTable.value, bcSenseTable.value, bcSyn0Sense.value, bcSyn1Sense.value)
          newSentenceIter += newSentence
        }
        newSentenceIter.toIterator
      }.cache()

      sentencesSplit(indexRDD).foreachPartition { sentencesIterator =>
        //println("foreachPartition ... ... ...")
        for (sentence <- sentencesIterator)
          learnParameters(sentence, bcHyperPara.value, bcExpTable.value, bcNegTable.value, bcSenseTable.value, bcSyn0Sense.value, bcSyn1Sense.value)
      }

      println("syn0Sense(0)(0)(0)=" + syn0Sense(0)(0)(0))
    }

    learned = true
  }

  //* sentence may need to be cloned
  private def adjustAssignment(sentence: Array[Int], hyperPara: mutable.HashMap[String, AnyVal], expTable: Array[Float], negTable: Array[Int], senseTable: Array[Int], syn0Sense: Array[Array[Array[Float]]], syn1Sense: Array[Array[Array[Float]]]): Array[Int] = {
    //println("adjustAssignment ... ... ...")

    for (wordPos <- 0 to sentence.size - 1) {

      require(hyperPara.get("numNeg").nonEmpty, "There is no hyperparameter \"numNeg\".")
      val numNeg = hyperPara.get("numNeg").get.asInstanceOf[Int]
      require(hyperPara.get("vocabSize").nonEmpty, "There is no hyperparameter \"vocabSize\".")
      val vocabSize = hyperPara.get("vocabSize").get.asInstanceOf[Int]
      require(hyperPara.get("window").nonEmpty, "There is no hyperparameter \"window\".")
      val window = hyperPara.get("window").get.asInstanceOf[Int]
      require(hyperPara.get("MAX_EXP").nonEmpty, "There is no hyperparameter \"MAX_EXP\".")
      val MAX_EXP = hyperPara.get("MAX_EXP").get.asInstanceOf[Int]

      val negSamples = getNegSamples(numNeg, negTable, senseTable, vocabSize)
      val neighbors = getNeighbors(sentence, wordPos, window)
      val word = sentence(wordPos) % vocabSize
      val bestSense = getBestSense(word, neighbors, negSamples, senseTable(word), syn0Sense, syn1Sense, expTable, MAX_EXP, vocabSize)

      sentence(wordPos) = word + bestSense * vocabSize
    }
    sentence
  }

  private def getNegSamples(numNeg: Int, negTable: Array[Int], senseTable: Array[Int], vocabSize: Int): Array[Int] = {
    //println("getNegSamples ... ... ...")

    val negSamples = new Array[Int](numNeg)
    val tableSize = negTable.size
    for (i <- 0 to numNeg - 1) {
      negSamples(i) = negTable(Math.abs(util.Random.nextLong() % tableSize).toInt)
      if (negSamples(i) <= 0)
        negSamples(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt
      //add sense information (assign sense randomly)
      negSamples(i) = negSamples(i) + vocabSize * util.Random.nextInt(senseTable(negSamples(i)))
    }
    negSamples
  }

  private def getNeighbors(sentence: Array[Int], wordPos: Int, window: Int): Array[Int] = {
    //println("getNeighbors ... ... ...")

    val neighbors = ArrayBuilder.make[Int]
    for (pos <- wordPos - window + 1 to wordPos + window - 1)
      if (pos >= 0 && pos < sentence.size && pos != wordPos)
        neighbors += sentence(pos)
    neighbors.result()
  }

  private def getBestSense(word: Int, neighbors: Array[Int], negSamples: Array[Int], numSenses: Int, syn0Sense: Array[Array[Array[Float]]], syn1Sense: Array[Array[Array[Float]]], expTable: Array[Float], MAX_EXP: Int, vocabSize: Int): Int = {
    //println("getBestSense ... ... ...")

    var bestSense = -1
    var bestScore = 0.0
    for (sense <- 0 to numSenses - 1) {
      //println("getBestSense ... ... ..."+ "word:" +word+" sense:"+sense)
      val score = getScore(word, sense, neighbors, negSamples, syn0Sense, syn1Sense, expTable, MAX_EXP, vocabSize)
      if (bestSense == -1 || score > bestScore) {
        bestScore = score
        bestSense = sense
      }
    }
    bestSense
  }

  private def getScore(word: Int, sense: Int, neighbors: Array[Int], negSamples: Array[Int], syn0Sense: Array[Array[Array[Float]]], syn1Sense: Array[Array[Array[Float]]], expTable: Array[Float], MAX_EXP: Int, vocabSize: Int): Double = {
    //println("getScore ... ... ...")

    var score = 0.0
    for (neighber <- neighbors) {

      val neighberWord = neighber % vocabSize
      val neighberSense = neighber / vocabSize
      //println("getScore ... ... ..."+"neighberWord:" +neighberWord+" neighberSense:"+neighberSense)
      score += math.log(activeFunction(syn1Sense(word)(sense), syn0Sense(neighberWord)(neighberSense), expTable, MAX_EXP))

      for (negSample <- negSamples) {
        val negWord = negSample % vocabSize
        val negSense = negSample / vocabSize
        //println("getScore ... ... ..."+"negWord:" +negWord+" negSense:"+negSense)
        if (negWord != word)
          score += math.log(1 - activeFunction(syn1Sense(negWord)(negSense), syn0Sense(neighberWord)(neighberSense), expTable, MAX_EXP))
      }
    }
    score
  }

  private def activeFunction(v0: Array[Float], v1: Array[Float], expTable: Array[Float], MAX_EXP: Int): Double = {
    //println("activeFunction ... ... ...")

    val vectorSize = v0.length
    var f = blas.sdot(vectorSize, v0, 1, v1, 1)
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

  //*
  private def learnParameters(sentence: Array[Int], hyperPara: mutable.HashMap[String, AnyVal], expTable: Array[Float], negTable: Array[Int], senseTable: Array[Int], syn0Sense: Array[Array[Array[Float]]], syn1Sense: Array[Array[Array[Float]]]): Unit = {
    //println("learnParameters ... ... ...")

    for (wordPos <- 0 to sentence.size - 1) {
      require(hyperPara.get("numNeg").nonEmpty, "There is no hyperparameter \"numNeg\".")
      val numNeg = hyperPara.get("numNeg").get.asInstanceOf[Int]
      require(hyperPara.get("vocabSize").nonEmpty, "There is no hyperparameter \"vocabSize\".")
      val vocabSize = hyperPara.get("vocabSize").get.asInstanceOf[Int]
      require(hyperPara.get("window").nonEmpty, "There is no hyperparameter \"window\".")
      val window = hyperPara.get("window").get.asInstanceOf[Int]
      require(hyperPara.get("MAX_EXP").nonEmpty, "There is no hyperparameter \"MAX_EXP\".")
      val MAX_EXP = hyperPara.get("MAX_EXP").get.asInstanceOf[Int]
      require(hyperPara.get("alpha").nonEmpty, "There is no hyperparameter \"alpha\".")
      val alpha = hyperPara.get("alpha").get.asInstanceOf[Float]

      val negSamples = getNegSamples(numNeg, negTable, senseTable, vocabSize)
      val neighbors = getNeighbors(sentence, wordPos, window)
      val wordWithSense = sentence(wordPos)

      learn(wordWithSense, neighbors, negSamples, syn0Sense, syn1Sense, expTable, MAX_EXP, vocabSize, alpha)
    }
  }

  private def learn(wordWithSense: Int, neighbors: Array[Int], negSamples: Array[Int], syn0Sense: Array[Array[Array[Float]]], syn1Sense: Array[Array[Array[Float]]], expTable: Array[Float], MAX_EXP: Int, vocabSize: Int, alpha: Float): Unit = {
    //println("learn ... ... ...")

    val word = wordWithSense % vocabSize
    val sense = wordWithSense / vocabSize
    //println("learn ... ... ..."+"word:" +word+" sense:"+sense)
    val vectorSize = syn0Sense(0)(0).length

    for (neighbor <- neighbors) {
      val neighberWord = neighbor % vocabSize
      val neighberSense = neighbor / vocabSize
      //println("learn ... ... ..."+"neighberWord:" +neighberWord+" neighberSense:"+neighberSense)
      val neu1e = new Array[Float](vectorSize)

      val g = (1 - activeFunction(syn1Sense(word)(sense), syn0Sense(neighberWord)(neighberSense), expTable, MAX_EXP)).toFloat * alpha
      blas.saxpy(vectorSize, g, syn1Sense(word)(sense), 1, neu1e, 1)
      blas.saxpy(vectorSize, g, syn0Sense(neighberWord)(neighberSense), 1, syn1Sense(word)(sense), 1)

      for (negSample <- negSamples) {

        val negWord = negSample % vocabSize
        val negSense = negSample / vocabSize
        //println("learn ... ... ..."+"negWord:" +negWord+" negSense:"+negSense)
        if (negWord != word) {

          val g = (-activeFunction(syn1Sense(negWord)(negSense), syn0Sense(neighberWord)(neighberSense), expTable, MAX_EXP)).toFloat * alpha
          blas.saxpy(vectorSize, g, syn1Sense(negWord)(negSense), 1, neu1e, 1)
          blas.saxpy(vectorSize, g, syn0Sense(neighberWord)(neighberSense), 1, syn1Sense(negWord)(negSense), 1)
        }

      }

      blas.saxpy(vectorSize, g, neu1e, 1, syn0Sense(neighberWord)(neighberSense), 1)
    }

  }

  def writeToFile(outputPath: String): Unit = {
    require(learned, "parameters need to be learned. You should learn parameters.")

    val file1 = new PrintWriter(new File(outputPath + "/wordSense.txt"))
    val file2 = new PrintWriter(new File(outputPath + "/syn0Sense.txt"))
    val file3 = new PrintWriter(new File(outputPath + "/syn1Sense.txt"))
    val file4 = new PrintWriter(new File(outputPath + "/vectorsSense.txt"))


    val wordIndex = vocabHash.toArray.sortWith((a, b) => a._2 < b._2)

    file4.write(vocabSize+" "+vectorSize+"\n")
    for ((wordString, word) <- wordIndex) {
      file4.write(wordString)
      for (sense <- 0 to senseTable(word) - 1) {
        file1.write(wordString + "_" + sense + "\n")
        //println(wordString + "_" + sense + " " + word)
        for (i <- 0 to vectorSize - 1)
          file4.write(" "+syn0Sense(word)(sense)(i))
        file4.write("\n")
      }
    }


    for (word <- 0 to vocabSize - 1) {
      for (sense <- 0 to senseTable(word) - 1) {
        for (i <- 0 to vectorSize - 1) {
          file2.write(syn0Sense(word)(sense)(i) + " ")
          file3.write(syn1Sense(word)(sense)(i) + " ")
        }
        file2.write("\n")
        file3.write("\n")
      }
    }

    file1.close()
    file2.close()
    file3.close()
    file4.close()
  }

}

object SenseAssignment {

  def loadModelSenses(path: String): Word2VecModel = {
    val wordIndex = collection.mutable.Map[String, Int]()
    var index = 0
    for (word <- Source.fromFile(path+"/wordSense.txt").getLines()) {
      wordIndex.put(word, index)
      index += 1
    }
    val wordVectors = Source.fromFile(path+"/syn0Sense.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s=>s.toFloat).toArray
    new Word2VecModel(wordIndex.toMap, wordVectors)
  }

}
