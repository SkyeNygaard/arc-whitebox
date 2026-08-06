#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef ORDER
#define ORDER 2047
#endif
#define DEPTH 32
static int N=ORDER;
typedef long double R;
static R *allocv(){ return (R*)calloc((size_t)N+1,sizeof(R)); }
static void zero(R*a){for(int i=0;i<=N;i++)a[i]=0;}
static void copyv(R*o,const R*a){for(int i=0;i<=N;i++)o[i]=a[i];}
static void mul(R*o,const R*a,const R*b){
  R*t=allocv();
  for(int n=0;n<=N;n++){
    R s=0;
    for(int i=0;i<=n;i++)s+=a[i]*b[n-i];
    t[n]=s;
  }
  copyv(o,t);free(t);
}
static void invs(R*o,const R*a){
  R*t=allocv();t[0]=1/a[0];
  for(int n=1;n<=N;n++){
    R s=0;for(int i=1;i<=n;i++)s+=a[i]*t[n-i];t[n]=-s/a[0];
  }copyv(o,t);free(t);
}
static void sqrts(R*o,const R*a){
  R*t=allocv();t[0]=sqrtl(a[0]);
  for(int n=1;n<=N;n++){
    R s=0;for(int i=1;i<n;i++)s+=t[i]*t[n-i];t[n]=(a[n]-s)/(2*t[0]);
  }copyv(o,t);free(t);
}
static void relu(R*out,const R*p){
 R *p2=allocv(),*om=allocv(),*root=allocv(),*dp=allocv(),*rinv=allocv(),*prod=allocv(),*ader=allocv(),*aser=allocv(),*mulv=allocv();
 mul(p2,p,p);om[0]=1-p2[0];for(int i=1;i<=N;i++)om[i]=-p2[i];sqrts(root,om);
 for(int i=0;i<N;i++)dp[i]=(i+1)*p[i+1];
 invs(rinv,root);mul(prod,dp,rinv);for(int i=0;i<=N;i++)ader[i]=-prod[i];
 aser[0]=acosl(p[0]);for(int i=1;i<=N;i++)aser[i]=ader[i-1]/i;
 R pi=acosl(-1.0L);
 for(int i=0;i<=N;i++){
   R pias=(i==0?pi-aser[0]:-aser[i]);
   // store pias temporarily in ader
   ader[i]=pias;
 }
 mul(mulv,ader,p);
 for(int i=0;i<=N;i++)out[i]=(root[i]+mulv[i])/pi;
 free(p2);free(om);free(root);free(dp);free(rinv);free(prod);free(ader);free(aser);free(mulv);
}

static void convsmall(R*out,const R*a,int na,const R*b,int nb){for(int i=0;i<=na+nb;i++)out[i]=0;for(int i=0;i<=na;i++)for(int j=0;j<=nb;j++)out[i+j]+=a[i]*b[j];}
int main(){
 R*k=allocv(),*tmpv=allocv();k[1]=1;
 for(int d=0;d<DEPTH;d++){zero(tmpv);relu(tmpv,k);R*sw=k;k=tmpv;tmpv=sw;}
 const R q[6]={
  9.747204751236081887459439233244991505630014158113377346536965157721415e-1L,
  2.77530920126995461395118350384323323719654413976321045348151188660926e-3L,
  2.417810485369684221599114225847104862649910243581530652492432628851725e-3L,
  1.809722082432843658457256941761114490489464383408042739885081415037772e-3L,
  1.529315320954953319047198695966819961039975102133967840958736312821237e-3L,
  1.23830589643062783981919946959869259350387403798801482689838750134661e-3L};
 const R P[4]={
 -2.570082998430774471979877792280289298383801553056559548532991951739213e-5L,
 -1.158575623550173439859797106173054971542012341868993832419582892612664e-2L,
  6.617723479576612630091590242028480805769918014799436974993737857558919e-3L,
  1.0L};
 R P2[7];convsmall(P2,P,3,P,3);
 R*rem=allocv();copyv(rem,k);for(int i=0;i<6;i++)rem[i]-=q[i];
 R*Q=allocv();
 for(int n=N;n>=6;n--){R cf=rem[n];Q[n-6]=cf;for(int j=0;j<7;j++)rem[n-6+j]-=cf*P2[j];}
 int QN=N-6;
 R*U=allocv();U[0]=sqrtl(Q[0]);
 for(int n=1;n<=QN;n++){R ss=0;for(int i=1;i<n;i++)ss+=U[i]*U[n-i];U[n]=(Q[n]-ss)/(2*U[0]);}
 R*L=allocv();for(int n=0;n<=QN+3;n++){R ss=0;for(int j=0;j<4;j++)if(n>=j&&n-j<=QN)ss+=P[j]*U[n-j];L[n]=ss;}
 R*B=allocv();B[0]=logl(Q[0]);
 for(int n=1;n<=QN;n++){R ss=0;for(int j=1;j<n;j++)ss+=(R)j*B[j]*Q[n-j];B[n]=((R)n*Q[n]-ss)/((R)n*Q[0]);}
 int negQ=0,negU=0,negL2=0,negB=0,incU=0;R minQ=1e999L,minU=1e999L,minL=1e999L,minB=1e999L;int iq=0,iu=0,il=0,ib=0;
 for(int n=0;n<=QN;n++){if(Q[n]<=0)negQ++;if(U[n]<=0)negU++;if(B[n]<=0&&n>0)negB++;if(Q[n]<minQ){minQ=Q[n];iq=n;}if(U[n]<minU){minU=U[n];iu=n;}if(n>0&&B[n]<minB){minB=B[n];ib=n;}if(n&&U[n]>U[n-1])incU++;}
 for(int n=2;n<=QN+3;n++){if(L[n]<=0)negL2++;if(L[n]<minL){minL=L[n];il=n;}}
 R maxrem=0;for(int i=0;i<6;i++)if(fabsl(rem[i])>maxrem)maxrem=fabsl(rem[i]);
 printf("ORDER %d QN %d maxrem %.9Le negQ %d negU %d negL2 %d negLog %d incU %d\n",N,QN,maxrem,negQ,negU,negL2,negB,incU);
 printf("mins Q[%d]=%.18Le U[%d]=%.18Le L[%d]=%.18Le log[%d]=%.18Le\n",iq,minQ,iu,minU,il,minL,ib,minB);
 for(int n=0;n<=QN;n++){if(n<20||n==100||n==500||n==1000||n==2000||n==4000||n==8000)printf("%d %.21Le %.21Le %.21Le %.21Le\n",n,Q[n],U[n],L[n],B[n]);}
 free(k);free(tmpv);free(rem);free(Q);free(U);free(L);free(B);return 0;
}
