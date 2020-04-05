import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ICubeResponse, IActionRequest, IScrambleRequest, IActionsResponse } from './rubiks/rubiks';

const urls = {
  solved: "http://127.0.0.1:5000/solved",
  action: "http://127.0.0.1:5000/action",
  scramble: "http://127.0.0.1:5000/scramble",
}

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  constructor(private http: HttpClient) { }

  public async getSolved() {
    return this.http.get<ICubeResponse>(urls.solved).toPromise();
  }

  public async performAction(actionRequest: IActionRequest) {
    return this.http.post<ICubeResponse>(urls.action, actionRequest).toPromise();
  }

  public async scramble(scrambleRequest: IScrambleRequest) {
    return this.http.post<IActionsResponse>(urls.scramble, scrambleRequest).toPromise();
  }
}
