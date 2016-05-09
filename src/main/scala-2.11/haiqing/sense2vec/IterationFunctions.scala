package haiqing.sense2vec

import com.github.fommil.netlib.BLAS.{getInstance => blas}

/**
 * Created by hwang on 04.03.16.
 */

//this class is used in the map function of spark
class IterationFunctions (private val window: Int, private val vectorSize: Int, private val multiSense: Int, private val negative: Int, private val vocabSize: Int,
                          private val senseCountAdd: Array[Array[Int]], private val senseCount: Array[Array[Int]],
                          private val expTable: Array[Float], private val senseTable: Array[Int], private val negTable: Array[Int],
                          private val syn0: Array[Array[Array[Float]]], private val syn1: Array[Array[Array[Float]]])  {
  private val MAX_EXP = 6
  private val U = 100
  private val ENCODE = 100

  private var sentence: Array[Int] = null
  private var sentenceNEG: Array[Array[Int]] = null

  //set sentence
  def setSentence(sentence: Array[Int]): Unit = {
    this.sentence = sentence
  }

  //set sentence negative samplings
  def setSentenceNEG(sentenceNEG: Array[Array[Int]]): Unit = {
    this.sentenceNEG = sentenceNEG
  }

  //generate sentence negative samples
  def generateSentenceNEG(): Unit = {
    sentenceNEG = sentence.map(w=>getNEG(w,negTable))
  }

  //adjust senses in the sentence
  def adjustSentence(): Boolean = {
    var flag = false
    for (pos <- 0 to sentence.size - 1) {
      val word = sentence(pos) / ENCODE
      if (senseTable(word) == multiSense) {
        var bestSense = -1 //there is no best sense
        var bestScore = 0.0
        for (sense <- 0 to senseTable(word) - 1) {
          val w = word * ENCODE + sense
          val score = getScore(w, pos)
          if (bestSense == -1 || score > bestScore) {
            bestScore = score
            bestSense = sense
          }
        }
        if (word * ENCODE + bestSense != sentence(pos)) {
          flag = true
        }
        sentence(pos) = word * ENCODE + bestSense
      }
    }
    flag
  }

  //loss = - score, skip-gram score from whole sentence
  def sentenceLoss(): Double = {
    var loss = 0.0
    for (pos <- 0 to sentence.size - 1)
      loss += -getScore(sentence(pos),pos)
    loss
  }

  //log probability of using center word to predict surrouding words
  private def getScore(w: Int, pos: Int): Double = {
    var score = 0.0
    for (p <- pos-window+1 to pos+window-1)
      if (p != pos && p >=0 && p < sentence.size) {
        val u = sentence(p)
        var NEG : Array[Int] = null
        if (sentenceNEG == null)
          NEG = getNEG(u,negTable)
        else
          NEG = sentenceNEG(p)
        score += Math.log(activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(u/ENCODE)(u%ENCODE)))
        for (z <- NEG)
          score += Math.log(1 - activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(z/ENCODE)(z%ENCODE)))
      }
    score
  }

  //count the sense which is used
  def addSenseCount(): Unit = {
    for (w <- sentence) {
      val word = w / ENCODE
      val sense = w % ENCODE
      senseCountAdd(word)(sense) += 1
    }
  }

  //skip-gram learning from the whole sentence
  def learnSentence(alpha: Float): (Double,Int) = {
    var loss = 0.0
    var lossNum = 0
    for (pos <- 0 to sentence.size - 1) {
      val w = sentence(pos)
      //val word = w / ENCODE
      //val sense = w % ENCODE
      //if (senseTable(word) == multiSense) {
        //if (multiSense == 1)
        //  learn(w, pos, alpha, alpha)
        //else
      val tmp = learn(w, pos, alpha, alpha)
      loss += tmp._1
      lossNum += tmp._2
        /*
        for (otherSense <- 0 to senseTable(word) - 1)
          if (sense != otherSense && senseCount(word)(sense) > senseCount(word)(otherSense) * U) {
            val w = word * ENCODE + otherSense
            learn(w, pos, alpha / 2, alpha / 2)
          }
          */
      //}
    }
    (loss,lossNum)
  }

  private def getNEG(w: Int, negTable: Array[Int]): Array[Int] = {
    val negSamples = new Array[Int](negative)
    val tableSize = negTable.size
    for (i <- 0 to negative - 1) {
      negSamples(i) = w
      while (negSamples(i)/ENCODE == w/ENCODE) {
        negSamples(i) = negTable(Math.abs(util.Random.nextLong() % tableSize).toInt)
        if (negSamples(i) <= 0)
          negSamples(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt
      }
      //add sense information (assign sense randomly)
      negSamples(i) = negSamples(i) * ENCODE + util.Random.nextInt(senseTable(negSamples(i)))
    }
    negSamples
  }
  
  //skip-gram model, use center word to predict surroung words
  private def learn(w: Int, pos: Int, alphaW: Float, alphaU: Float): (Double,Int) = {
    var loss = 0.0
    var lossNum = 0
    for (p <- pos-window+1 to pos+window-1)
      if (p != pos && p >=0 && p < sentence.size) {
        val u = sentence(p)
        var NEG : Array[Int] = null
        if (sentenceNEG == null)
          NEG = getNEG(u,negTable)
        else
          NEG = sentenceNEG(p)
        val gradientW = new Array[Float](vectorSize)
        val l = activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(u/ENCODE)(u%ENCODE))
        val g = (1-l).toFloat
        loss += -math.log(l)
        lossNum += 1
        blas.saxpy(vectorSize, g, syn1(u/ENCODE)(u%ENCODE), 1, gradientW, 1)
        blas.saxpy(vectorSize, g*alphaU, syn0(w/ENCODE)(w%ENCODE), 1, syn1(u/ENCODE)(u%ENCODE), 1)
        for (z <- NEG) {
          val l = activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(z/ENCODE)(z%ENCODE))
          val g = (-l).toFloat
          loss += -math.log(1-l)
          lossNum += 1
          blas.saxpy(vectorSize, g, syn1(z/ENCODE)(z%ENCODE), 1, gradientW, 1)
          blas.saxpy(vectorSize, g*alphaU, syn0(w/ENCODE)(w%ENCODE), 1, syn1(z/ENCODE)(z%ENCODE), 1)
        }
        blas.saxpy(vectorSize, alphaW, gradientW, 1, syn0(w/ENCODE)(w%ENCODE), 1)
    }
    (loss,lossNum)
  }

  //sigmoid function applying on two vectors
  private def activeFunction(v0: Array[Float], v1: Array[Float]): Double = {
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
}
