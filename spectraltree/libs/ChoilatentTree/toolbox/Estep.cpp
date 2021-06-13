/* Computes the expected statistics on a tree 
 * (i.e., node and edge marginals) with hidden variables 
 * given the potentials.  Also computes the entropy of the posterior
 * distribution over hidden variables.
 * Myung Jin Choi, August 2010.  MIT.
 */


#include <math.h>
#include "mex.h"
#include <vector>
#include <iostream>

using namespace std;

extern void _main();

int numStates,numNodes,numObserved,numSamples,numEdges;
int rootNodeNum;
static const double tiny=1.0E-20;
bool verbose = false;
      
vector<double> operator +(const vector<double>& v1, const vector<double>& v2)
{
    vector<double> result;
    for (int i=0; i<v1.size(); i++)
    {
        result.push_back(v1[i]+v2[i]);        
    }    
    return result;
}

vector<double> operator *(const vector<double>& v1, const vector<double>& v2)
{
    vector<double> result;
    for (int i=0; i<v1.size(); i++)
    {
        result.push_back(v1[i]*v2[i]);        
    }    
    return result;
}

vector<double> operator /(const vector<double>& v1, const vector<double>& v2)
{
    vector<double> result;
    for (int i=0; i<v1.size(); i++)
    {
        if(v2[i]==0)
            result.push_back(v1[i]); 
        else
            result.push_back(v1[i]/v2[i]);        
    }    
    return result;
}


double dotProduct(const vector<double>& v1, const vector<double>& v2)
{
    double sum = 0.0;
    for(int i=0; i<v1.size(); i++)
    {
        sum += v1[i]*v2[i];
    }
    return sum;
}

vector<double> vectorMatrixProduct(const vector<double>& v, const vector<double>& m)
{
    vector<double> result;
    int m_ind = 0;    
    for(int i=0; i<v.size(); i++)
    { 
        double sum = 0.0;
        for(int j=0; j<v.size(); j++)
        {
            sum += v[j]*m[m_ind];
            m_ind++;
        }            
        result.push_back(sum);
    }
    return result;
}

vector<double> matrixVectorProduct(const vector<double>& m, const vector<double>& v)
{
    vector<double> result(v.size());
    int m_ind = 0;    
    for(int i=0; i<v.size(); i++)
    { 
        for(int j=0; j<v.size(); j++)
        {
            result[j] += v[i]*m[m_ind];
            m_ind++;
        }     
    }
    return result;
}

double elementSum(const vector<double>& v1)
{
    double sum = 0.0;
    for(int i=0; i<v1.size(); i++)
    {
        sum += v1[i];
    }
    return sum;
}

vector<double> normalize(const vector<double>& v1)
{
    vector<double> result = v1;
    double sum = elementSum(v1);
    if(sum > 0.0)
    {
        for (int i=0; i<result.size(); i++)
        {
            result[i] /= sum;
        }
    }
    return result;
}

double vDotProductLogv(const vector<double>& v1)
{
    double result = 0.0;
    for(int i=0; i<v1.size(); i++)
    {
        result += v1[i]*log(v1[i]+tiny);
    }
    return result;
}

vector<vector<double> > operator +(const vector<vector<double> >& v1, const vector<vector<double> >& v2)
{
    vector<vector <double> >result;
    for (int i=0; i<v1.size(); i++)
    {
        result.push_back(v1[i]+v2[i]);
    }
    return result;
}

class Node{
public:
    Node(){}
    Node(int n, vector<double> np, bool is_obs, int deg)
    :nodeNum(n), node_potential(np), is_observed(is_obs), hnodeDegree(deg)
    {   
        //cml_node_marginal.resize(numStates);
    }
    int getNodeNumber() { return nodeNum;}
    vector<double> getNodePotential() { return node_potential; }
    vector<double> getProdInMsgs(){ return prod_incoming_messages; }
    bool isObserved() {return is_observed; }
    int getDegree() {return hnodeDegree; }
    //vector<double> getCumulativeNodeMarginal() {return cml_node_marginal; }
    void setNodeMarginal(vector<double> nm)    
    { 
        node_marginal = nm; 
        //cml_node_marginal =cml_node_marginal+node_marginal;
    }
    void updateProdInMsgs(vector<double> msg)
    {
        prod_incoming_messages = prod_incoming_messages*normalize(msg);
        //prod_incoming_messages = normalize(prod_incoming_messages*msg);
    }
    void initMsg()
    {
        prod_incoming_messages.clear();
        prod_incoming_messages.resize(numStates,1.0);
    }
private:
    int nodeNum;
    int hnodeDegree;
    bool is_observed;
    vector<double> node_potential;
    vector<double> node_marginal;
    //vector<double> cml_node_marginal;
    vector<double> prod_incoming_messages;
};

class Edge{
public:
    Edge(int p, int c, vector <double> cp)
    :parentNum(p), childNum(c), cond_prob(cp)
    {    
        cml_edge_marginal.resize(numStates*numStates,0.0);
    }
    Edge(){ }
    int getParent(){ return parentNum; }
    int getChild(){ return childNum; }
    vector<double> getEdgePotential() {return cond_prob; }
    vector<double> getChild2ParentMsg() {return msg_c2p; }
    vector<double> getCumulativeEdgeMarginal() {return cml_edge_marginal; }
    void setEdgeMarginal(vector <double> nm)    
    { 
        edge_marginal = nm; 
        cml_edge_marginal =cml_edge_marginal+edge_marginal;
    }    
    void setChild2ParentMsg(vector<double> msg) {msg_c2p = msg;}

private:
    int parentNum;
    int childNum;
    vector<double> cond_prob;
    vector<double> edge_marginal;
    vector<double> cml_edge_marginal;
    vector<double> msg_c2p;
};


vector<Node> setNodePotentials(double* nodePotential, double *degree_hnodes)
{
    vector<Node> nodes(numNodes);
    vector<double> nodePot(numStates);
    for (int n=0; n<numNodes; n++)
    {
        bool isObserved = true;
        int degree = 0;
        if(n >= numObserved)
        {
            isObserved = false;
            degree = (int) degree_hnodes[n-numObserved];
        }        
        for(int k=0; k<numStates; k++)
        {
            nodePot[k] = nodePotential[numStates*n+k];
        }                    
        nodes[n] = Node(n, nodePot, isObserved, degree);
    }
    return nodes;
}

vector<Edge> setEdgePotentials(double* edgePotential, double* treeMsgOrder)
{
    vector<Edge> edges(numEdges);
    vector<double> edgePot(numStates*numStates);
    for (int e=0; e<numEdges; e++)
    {
        int p = (int) treeMsgOrder[2*(numEdges+e)]-1;
        int c = (int) treeMsgOrder[2*(numEdges+e)+1]-1;
        for (int sc=0; sc<numStates; sc++)
        {
            for (int sp=0; sp<numStates; sp++)
            {
                int row_ind = numStates*p+sp;
                int col_ind = numStates*c+sc;
                int num_rows = numStates*numNodes;
                edgePot[numStates*sc+sp] = edgePotential[num_rows*col_ind+row_ind];
                //mexPrintf("p = %d, c = %d, %g\n", p,c,edgePot[sp][sc]);
            }
        }        
        edges[e] = Edge(p,c,edgePot);       
    }
    return edges;    
}

vector<double> getParentMarginal(vector<double> edgeMarginal)
{
    vector<double> pmarginal(numStates);
    for(int sc=0; sc<numStates; sc++)
    {
        for(int sp=0; sp<numStates; sp++)
        {
            pmarginal[sp] += edgeMarginal[numStates*sc+sp];
        }
    }
    return pmarginal;
}

vector<double> getChildMarginal(vector<double> edgeMarginal)
{
    vector<double> cmarginal(numStates);
    for(int sc=0; sc<numStates; sc++)
    {
        for(int sp=0; sp<numStates; sp++)
        {
            cmarginal[sc] += edgeMarginal[numStates*sc+sp];
        }
    }
    return cmarginal;
}

double sumProduct(double *sample, vector<Node>& nodes, vector<Edge>& edges)
{
    double h_entropy = 0.0;
    for(int i=0; i<nodes.size(); i++)  nodes[i].initMsg();       

    Node parent, child;
    // Upward pass. Passing messages from child to parent
    for (int e=edges.size()-1; e>=0; e--)
    {
        parent = nodes[edges[e].getParent()];
        child = nodes[edges[e].getChild()];
        vector<double> msg(numStates);
        if(!parent.isObserved())
        {            
            if(child.isObserved())
            {
                msg[(int) sample[child.getNodeNumber()]] = 1;
            }
            else
            {
                msg = child.getProdInMsgs();
            }            
            //msg = vectorMatrixProduct(msg,edges[e].getEdgePotential());            
            msg = matrixVectorProduct(edges[e].getEdgePotential(),msg);   
            nodes[edges[e].getParent()].updateProdInMsgs(msg);       
            edges[e].setChild2ParentMsg(msg);
        }   
        if(verbose)
        {
            mexPrintf("\nPassing messages from %d to %d\n",child.getNodeNumber(), parent.getNodeNumber());
            for (int i=0; i<numStates; i++) mexPrintf("%g ",msg[i]);
        }
    }
    
    // Downward pass.  Passing messages from parent to child
    // Compute edge marginals and the entropy for hidden variables.
    for(int e=0; e<edges.size(); e++)
    {
        vector<double> msg(numStates);
        vector<double> emar(numStates*numStates);
        vector<double> ePot = edges[e].getEdgePotential();
        parent = nodes[edges[e].getParent()];
        child = nodes[edges[e].getChild()];
         
        vector<double> parentProdMsg(numStates);
        vector<double> childProdMsg(numStates);
        int pSample, cSample;
        if(child.isObserved())
        {
            cSample = (int) sample[child.getNodeNumber()];   
            childProdMsg[cSample] = 1;
        }
        else
        {
            childProdMsg = child.getProdInMsgs();
        }
            
        if(parent.isObserved())
        {
            pSample = (int) sample[parent.getNodeNumber()];
            parentProdMsg[pSample] = 1;
        }
        else 
        {
            parentProdMsg = parent.getProdInMsgs();
            if(parent.getNodeNumber()==rootNodeNum)
            {
                parentProdMsg = parentProdMsg*parent.getNodePotential();
            }
            parentProdMsg = parentProdMsg/edges[e].getChild2ParentMsg();
        }
        
        // compute edge marginals
        for(int sc=0; sc<numStates; sc++)
        {
            for(int sp=0; sp<numStates; sp++)
            {                    
                emar[numStates*sc+sp] = ePot[numStates*sc+sp]*childProdMsg[sc]*parentProdMsg[sp];
            }                   
        }        
        emar = normalize(emar);
        edges[e].setEdgeMarginal(emar);
        
        // Pass message from parent to child and compute the entropy;
        if(!child.isObserved())
        {
            msg = vectorMatrixProduct(parentProdMsg,ePot);
            nodes[edges[e].getChild()].updateProdInMsgs(msg);
            if(!parent.isObserved())
            {
                h_entropy -= vDotProductLogv(emar);
                //mexPrintf("Edge entropy between %d and %d is %g\n",parent.getNodeNumber(),child.getNodeNumber(),-vDotProductLogv(emar));

            }
            h_entropy += (child.getDegree()-1)*vDotProductLogv(getChildMarginal(emar));
            //mexPrintf("Node entropy of %d with degree %d is %g\n",child.getNodeNumber(),child.getDegree(),(child.getDegree()-1)*vDotProductLogv(getChildMarginal(emar)));
            
        }     
        if(!parent.isObserved() && e==0) // entropy for the root node if it is hidden
            h_entropy += (parent.getDegree()-1)*vDotProductLogv(getParentMarginal(emar));            

        
        if(verbose)
        {
            mexPrintf("\nPassing messages from %d to %d\n",parent.getNodeNumber(), child.getNodeNumber());
            for (int i=0; i<numStates; i++) mexPrintf("%g ",msg[i]);
            mexPrintf("\nEdge marginals:");
            for (int i=0; i<numStates*numStates; i++) mexPrintf("%g ",emar[i]);        
        }
    }    
    return h_entropy;    
}

double runEstep(double *samples, vector<Node>& nodes, vector<Edge>& edges)
{
    double h_entropy = 0.0;
    double *samplesn = &samples[0];
    for(int m=0; m<numSamples; m++)
    {
        h_entropy += sumProduct(samplesn,nodes,edges);
        //mexPrintf("%g\n",h_entropy);
        samplesn+=numObserved;        
    }
    
    return h_entropy;
}

void writeEdgeMarginals(double* edge_marginals, vector<Edge> edges)
{     
     int num_rows = numStates*numNodes;
     int col_ind,row_ind,p,c;
     vector<double> cum_emar;
     vector<double> cum_nmar;
     for(int e=0; e<edges.size(); e++)
     {
         cum_emar = edges[e].getCumulativeEdgeMarginal();
         cum_nmar = getChildMarginal(cum_emar);
         p = edges[e].getParent();
         c = edges[e].getChild();
         for (int sc=0; sc<numStates; sc++)
         {
            col_ind = numStates*c+sc;
            for (int sp=0; sp<numStates; sp++)
            {
                row_ind = numStates*p+sp;
                edge_marginals[num_rows*col_ind+row_ind] = cum_emar[numStates*sc+sp]/numSamples;
            }
            edge_marginals[num_rows*col_ind+col_ind] = cum_nmar[sc]/numSamples;
         }               
     }  
     cum_emar = edges[0].getCumulativeEdgeMarginal();
     cum_nmar = getParentMarginal(cum_emar);
     p = edges[0].getParent();
     for (int sp=0; sp<numStates; sp++)
     {
            row_ind = numStates*p+sp;
            edge_marginals[num_rows*row_ind+row_ind] = cum_nmar[sp]/numSamples;
     }

     return;
     
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{

  /* Check for proper number of arguments */

  if (nrhs != 5) {
    mexErrMsgTxt("Estep requires five input arguments.");
  } else if (nlhs != 2) {
    mexErrMsgTxt("MEXCPP requires two output arguments.");
  }
    double *samples, *nodePotential, *edgePotential, *treeMsgOrder, *degreeHnodes;
    samples = mxGetPr(prhs[0]);
    nodePotential = mxGetPr(prhs[1]);
    edgePotential = mxGetPr(prhs[2]);
    treeMsgOrder = mxGetPr(prhs[3]);
    degreeHnodes = mxGetPr(prhs[4]);

    numObserved = mxGetM(prhs[0]);
    numSamples = mxGetN(prhs[0]);
    numStates = mxGetM(prhs[1]);
    numNodes = mxGetN(prhs[1]);
    numEdges = numNodes-1;
    
    vector<Node> nodes = setNodePotentials(nodePotential,degreeHnodes);
    vector<Edge> edges = setEdgePotentials(edgePotential,treeMsgOrder);
    rootNodeNum = edges[0].getParent();    
    
    /* Create an mxArray for the output data */
    double *edge_marginals;
    plhs[0] = mxCreateDoubleMatrix(numNodes*numStates, numNodes*numStates, mxREAL);
    edge_marginals = mxGetPr(plhs[0]);
    
    double *h_entropy;
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    h_entropy = mxGetPr(plhs[1]);
    h_entropy[0] = runEstep(samples,nodes,edges);
    if(verbose)
        mexPrintf("%g\n",h_entropy[0]);
    
    
    writeEdgeMarginals(edge_marginals,edges);    

    return;
}

